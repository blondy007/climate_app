"""Per-browser appearance preferences, separate from the shared forecast store."""
import json

import streamlit as st
from streamlit.components.v1 import html


THEMES = {"auto": "Automático", "light": "Claro", "dark": "Oscuro"}
PALETTES = {
    "light": {"background": "#f3f6f7", "surface": "#ffffff", "ink": "#193543",
              "muted": "#617783", "line": "#e0e9ec", "grid": "#edf2f4"},
    "dark": {"background": "#101e27", "surface": "#192d38", "ink": "#e6f0f4",
             "muted": "#a3bac6", "line": "#304854", "grid": "#30434e"},
}


def sync_appearance():
    """Bridge browser storage/media preferences through a native Streamlit radio.

    The hidden radio uses the regular widget callback, so restoring a preference
    doesn't reload the page, discard a form or reset an authenticated session.
    Explicit URL choices win over browser storage. CSS also follows device changes.
    """
    state = st.session_state
    requested = st.query_params.get("theme", state.get("appearance", "auto"))
    mode = requested if requested in THEMES else "auto"
    scheme = st.query_params.get("scheme", state.get("appearance_resolved", "light"))
    state["appearance"] = mode
    state["appearance_resolved"] = scheme if scheme in PALETTES else "light"
    def browser_changed():
        chosen = state["appearance_sync"]
        state["appearance"] = "auto" if chosen.startswith("auto-") else chosen
        state["appearance_resolved"] = chosen.removeprefix("auto-")
        st.query_params["theme"] = state["appearance"]
        st.query_params["scheme"] = state["appearance_resolved"]

    current = f'auto-{state["appearance_resolved"]}' if mode == "auto" else mode
    state["appearance_sync"] = current
    st.markdown('<span class="appearance-sync-anchor"></span>', unsafe_allow_html=True)
    st.radio("Sincronizar apariencia", ["auto-light", "auto-dark", "light", "dark"],
             key="appearance_sync", on_change=browser_changed, label_visibility="collapsed")
    args = json.dumps({"mode": mode, "current": current})
    html('''<script>(()=>{
      const args = ''' + args + ''';
      const root = window.parent;
      const explicit = new URL(root.location.href).searchParams.has("theme");
      const storageKey = "climateapp.appearance.v1";
      const media = root.matchMedia("(prefers-color-scheme: dark)");
      const select = chosen => {
        const anchor = root.document.querySelector(".appearance-sync-anchor");
        const control = anchor?.closest(".element-container")?.nextElementSibling;
        const label = [...(control?.querySelectorAll("label") || [])].find(el => el.textContent.trim() === chosen);
        const input = label?.querySelector("input");
        if (input && !input.checked) input.click();
      };
      const first = !root.climateAppAppearanceReady;
      root.climateAppAppearanceReady = true;
      let mode = args.mode;
      try {
        const saved = root.localStorage.getItem(storageKey);
        if (first && !explicit && ["auto","light","dark"].includes(saved)) mode = saved;
        if (root.climateAppAppearanceMode !== mode) root.localStorage.setItem(storageKey, mode);
      } catch (_) { /* Session and URL preferences work when storage is blocked. */ }
      root.climateAppAppearanceMode = mode;
      const chosen = mode === "auto" ? `auto-${media.matches ? "dark" : "light"}` : mode;
      if (chosen !== args.current) select(chosen);
      if (root.climateAppAppearanceListener) media.removeEventListener("change", root.climateAppAppearanceListener);
      root.climateAppAppearanceListener = () => { if (mode === "auto") select(`auto-${media.matches ? "dark" : "light"}`); };
      media.addEventListener("change", root.climateAppAppearanceListener);
    })();</script>''', height=0)
    return mode


def palette():
    mode = st.session_state.get("appearance", "auto")
    resolved = st.session_state.get("appearance_resolved", "light") if mode == "auto" else mode
    return PALETTES.get(resolved, PALETTES["light"])


def appearance_controls():
    def remember():
        mode = st.session_state["appearance_picker"]
        st.session_state["appearance"] = mode
        st.query_params["theme"] = mode

    st.session_state["appearance_picker"] = st.session_state.get("appearance", "auto")
    st.radio("Tema de la aplicación", list(THEMES), format_func=THEMES.get, horizontal=True,
             key="appearance_picker", on_change=remember, label_visibility="collapsed")
    st.caption("Automático sigue el tema de tu móvil u ordenador. Tu elección se recuerda en este navegador.")
