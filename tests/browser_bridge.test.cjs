const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync('climate_app/ui/browser_component/index.html', 'utf8').match(/<script>([\s\S]*)<\/script>/)[1];

function harness({saved, denied = false, secure = true, deferred = false, response} = {}) {
  const messages = [], callbacks = {}, storage = new Map(), positions = [], requests = [];
  if (saved) storage.set('climateapp.forecast.preferences.v1', JSON.stringify(saved));
  let now = 1_800_000_000_000, calls = 0, timerId = 0;
  const intervals = new Map();
  const parent = {postMessage: message => messages.push(message)};
  const position = {coords: {latitude: 37.744126, longitude: -.850456, accuracy: 15}, timestamp: now};
  const document = {visibilityState: 'visible', addEventListener: (name, fn) => callbacks[name] = fn};
  const window = {parent, isSecureContext: secure, addEventListener: (name, fn) => callbacks[name] = fn};
  vm.runInNewContext(source, {window, document, Intl,
    Date: class extends Date {static now() {return now;}},
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    AbortController, setTimeout, clearTimeout,
    fetch: async (url, options) => {
      requests.push({url, options});
      const data = response ? await response(url, options) : url.startsWith('https://api.bigdatacloud.net/')
        ? {locality: 'Los Narejos', city: 'Los Alcázares'} : {authenticated: false, csrf: 'fixture-csrf'};
      return {ok: true, json: async () => data};
    },
    navigator: {geolocation: {getCurrentPosition: (success, error) => {
      calls++; positions.push(success);
      if (!deferred) denied ? error({code: 1}) : success({...position, timestamp: now});
    }}}, setInterval: fn => {intervals.set(++timerId, fn); return timerId;},
    clearInterval: id => intervals.delete(id)});
  return {messages, storage, positions, document, parent, requests,
    render: args => callbacks.message({source: parent, data: {type: 'streamlit:render', args}}),
    event: (source, args) => callbacks.message({source, data: {type: 'streamlit:render', args}}),
    advance: ms => {now += ms; [...intervals.values()].forEach(fn => fn());}, calls: () => calls,
    values: () => messages.filter(m => m.type === 'streamlit:setComponentValue').at(-1)?.value.events || []};
}
const active = {automatic: true, weather_view: true, restore: false, request: 0, model: 'auto'};

test('retries readiness until the parent acknowledges a render', () => {
  const h = harness();
  const readyCount = () => h.messages.filter(m => m.type === 'streamlit:componentReady').length;
  assert.equal(readyCount(), 1);
  h.advance(250); assert.equal(readyCount(), 2);
  h.render({...active, automatic: false});
  h.advance(250); assert.equal(readyCount(), 2);
  assert.equal(h.messages.find(m => m.type === 'streamlit:setFrameHeight').height, 0);
});

test('restores preferences before requesting permission', () => {
  const h = harness({saved: {automatic: true, model: 'icon'}});
  h.render({...active, automatic: false, restore: true});
  assert.equal(h.calls(), 0);
  assert.equal(h.values()[0].kind, 'preferences');
  assert.equal(h.values()[0].model, 'icon');
  assert.equal(h.values()[0].automatic, true);
});
test('coordinates are coarse before transmission and never enter storage', () => {
  const h = harness(); h.render(active);
  assert.equal(h.values()[0].latitude, 37.74);
  assert.equal(h.values()[0].longitude, -.85);
  assert.equal(h.values()[0].kind, 'position');
  const saved = JSON.parse([...h.storage.values()][0]);
  assert.deepEqual(Object.keys(saved).sort(), ['automatic', 'model', 'place_id']);
});
test('rerenders do not repeatedly fetch GPS; visible automatic mode refreshes after five minutes', () => {
  const h = harness(); h.render(active); h.render({...active, model: 'icon'});
  assert.equal(h.calls(), 1);
  h.advance(300000); assert.equal(h.calls(), 2);
  h.render({...active, weather_view: false}); h.advance(300000);
  assert.equal(h.calls(), 2);
});
test('denied permission is not retried until an explicit retry', () => {
  const h = harness({denied: true}); h.render(active); h.advance(600000);
  assert.equal(h.calls(), 1); assert.equal(h.values()[0].reason, 'denied');
  h.render({...active, request: 1}); assert.equal(h.calls(), 2);
});
test('an outstanding location cannot override a subsequently chosen saved place', () => {
  const h = harness({deferred: true}); h.render(active);
  h.render({...active, automatic: false});
  h.positions[0]({coords: {latitude: 40, longitude: -3, accuracy: 10}, timestamp: Date.now()});
  assert.equal(h.values().length, 0);
});
test('insecure contexts fail closed and unrelated messages are ignored', () => {
  const h = harness({secure: false}); h.event({}, active); assert.equal(h.values().length, 0);
  h.render(active); assert.equal(h.values()[0].reason, 'https'); assert.equal(h.calls(), 0);
});

const flush = () => new Promise(resolve => setImmediate(resolve));

test('saved place survives a fresh session without an early default overwrite or GPS request', () => {
  const id = '0123456789abcdef';
  const h = harness({saved: {automatic: false, place_id: id, model: 'icon'}});
  h.render({...active, automatic: false, restore: true});
  assert.equal(h.values()[0].place_id, id);
  h.render({...active, automatic: false, restore: true});
  assert.equal(JSON.parse([...h.storage.values()][0]).place_id, id);
  h.render({...active, automatic: false, restore: false, place_id: id, model: 'icon'});
  const reopened = harness({saved: JSON.parse([...h.storage.values()][0])});
  reopened.render({...active, automatic: false, restore: true});
  assert.equal(reopened.values()[0].place_id, id);
  assert.equal(reopened.calls(), 0);
});

test('automatic place name uses coarse coordinates, follows the position and stays out of storage', async () => {
  const h = harness(); h.render(active); await flush();
  const values = h.values();
  assert.ok(values.findIndex(e => e.kind === 'position') < values.findIndex(e => e.kind === 'place_name'));
  assert.equal(values.find(e => e.kind === 'place_name').name, 'Los Narejos · Los Alcázares');
  const request = h.requests.find(r => r.url.includes('reverse-geocode-client'));
  assert.match(request.url, /latitude=37\.74&longitude=-0\.85&/);
  assert.equal(request.options.credentials, 'omit');
  const stored = [...h.storage.values()].join();
  assert.ok(!stored.includes('37.74') && !stored.includes('Narejos'));
  h.advance(300000); await flush();
  assert.equal(h.requests.filter(r => r.url.includes('reverse-geocode-client')).length, 1);
});

test('failed or late place names do not replace the selected saved place', async () => {
  let finish;
  const h = harness({response: url => url.startsWith('https:') ? new Promise(resolve => finish = resolve)
    : {authenticated: false, csrf: 'fixture-csrf'}});
  h.render(active);
  h.render({...active, automatic: false, place_id: '0123456789abcdef'});
  finish({locality: 'Late city'}); await flush();
  assert.equal(h.values().filter(e => e.kind === 'place_name').length, 0);
  const failed = harness({response: async url => {if (url.startsWith('https:')) throw new Error('offline');
    return {authenticated: false, csrf: 'fixture-csrf'};}});
  failed.render(active); await flush();
  assert.ok(failed.values().some(e => e.kind === 'position'));
  assert.ok(!failed.values().some(e => e.kind === 'place_name'));
  const ip = harness({response: async url => url.startsWith('https:')
    ? {lookupSource: 'ipGeolocation', city: 'IP city'} : {authenticated: false, csrf: 'fixture-csrf'}});
  ip.render(active); await flush();
  assert.ok(!ip.values().some(e => e.kind === 'place_name'));
});
test('concurrent preference and auth replies are delivered together without storing credentials', async () => {
  const h = harness({saved: {model: 'icon'}, response: () => ({authenticated: true, proof: 'one-use-proof'})});
  h.render({...active, automatic: false, restore: true});
  await flush();
  assert.deepEqual(Array.from(h.values(), e => e.kind), ['preferences', 'auth']);
  assert.equal([...h.storage.values()].some(v => v.includes('proof')), false);
});
test('remember and logout requests are serialized even when the remember response is delayed', async () => {
  let finishRemember;
  const h = harness({response: (url) => url.endsWith('/remember') ? new Promise(resolve => finishRemember = resolve)
    : {authenticated: false, csrf: 'fixture-csrf'}});
  h.render({...active, automatic: false, auth_action: {kind: 'remember', code: 'fixture-code', nonce: 'login'}});
  await flush();
  h.render({...active, automatic: false, auth_action: {kind: 'logout', nonce: 'logout'}});
  await flush();
  assert.deepEqual(h.requests.map(r => r.url.split('/').at(-1)), ['session', 'remember']);
  finishRemember({authenticated: true, proof: 'fixture-proof'});
  await flush();
  assert.deepEqual(h.requests.map(r => r.url.split('/').at(-1)), ['session', 'remember', 'session', 'logout']);
  assert.deepEqual(Array.from(h.values(), e => e.kind), ['auth', 'auth_logout']);
  const remember = h.requests[1].options;
  assert.equal(remember.headers['X-Xsrftoken'], 'fixture-csrf');
  assert.equal(remember.credentials, 'same-origin');
});
