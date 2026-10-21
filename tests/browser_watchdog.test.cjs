const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const source = readFileSync('climate_app/ui/browser_component/watchdog.html', 'utf8').match(/<script>([\s\S]*)<\/script>/)[1];
const asset = 'https://weather.example/component/climate_app.ui.browser_location.climateapp_browser_0123456789/index.html';
function harness({src=asset, documentURL='about:blank', response}={}) {
  let callback, requests=0;
  const frame = {src, contentDocument:{URL:documentURL}, isConnected:true};
  const root = {document:{querySelector:()=>frame}, location:new URL('https://weather.example/'),
    fetch:async (url, options)=>{requests++; assert.equal(options.credentials,'same-origin');
      assert.equal(options.redirect,'error'); assert.equal(url,src);
      if (response) await response(frame);
      return {ok:true,headers:{get:()=> 'text/html'},text:async()=>'<html>first-party fixture</html>'};}};
  vm.runInNewContext(source,{window:{parent:root},URL,setTimeout:fn=>callback=fn});
  return {frame,root,check:()=>callback(),requests:()=>requests};
}
test('recovers only an empty first-party bridge document without weakening its sandbox',async()=>{
  const h=harness(); await h.check();
  assert.equal(h.requests(),1); assert.equal(h.frame.srcdoc,'<html>first-party fixture</html>');
});
test('does not replace an active document or interrupt a pending permission or login',async()=>{
  const h=harness({documentURL:asset}); await h.check(); assert.equal(h.requests(),0);
  const late=harness({response:frame=>frame.contentDocument.URL=asset}); await late.check();
  assert.equal(late.frame.srcdoc,undefined);
});
test('refuses external URLs and unrelated first-party content',async()=>{
  for (const src of ['https://other.example/component/climate_app.ui.browser_location.climateapp_browser_0123456789/index.html',
                     'https://weather.example/_climate/auth/session','https://weather.example/component/unknown/index.html']) {
    const h=harness({src}); await h.check(); assert.equal(h.requests(),0);
  }
});
