// NeoMerger — JS helpers
// Populates hidden fields with txt2img settings BEFORE the click event fires.
// This avoids the unreliable _js / js= callback on different Gradio versions.

function neomerger_get_value(elem_id, fallback) {
    const el = gradioApp().querySelector('#' + elem_id);
    if (!el) return fallback;
    const ta = el.querySelector('textarea') || el.querySelector('input');
    if (ta) return ta.value;
    return fallback;
}

function neomerger_get_dropdown(elem_id, fallback) {
    const el = gradioApp().querySelector('#' + elem_id);
    if (!el) return fallback;
    const input = el.querySelector('input');
    if (input) return input.value;
    return fallback;
}

function neomerger_set_field(elem_id, value) {
    const el = gradioApp().querySelector('#' + elem_id);
    if (!el) {
        console.warn('[NeoMerger] field not found: ' + elem_id);
        return;
    }
    const target = el.querySelector('textarea') || el.querySelector('input');
    if (!target) {
        console.warn('[NeoMerger] no textarea/input inside ' + elem_id);
        return;
    }
    const proto = target instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
    setter.call(target, value);
    target.dispatchEvent(new Event('input',  { bubbles: true }));
    target.dispatchEvent(new Event('change', { bubbles: true }));
}

// Reads the live txt2img settings once. Returns an object.
function neomerger_read_t2i() {
    let seed = parseFloat(neomerger_get_value('txt2img_seed', -1));
    if (isNaN(seed)) seed = -1;
    return {
        prompt:  neomerger_get_value('txt2img_prompt', ''),
        neg:     neomerger_get_value('txt2img_neg_prompt', ''),
        steps:   parseFloat(neomerger_get_value('txt2img_steps', 28)) || 28,
        cfg:     parseFloat(neomerger_get_value('txt2img_cfg_scale', 7)) || 7,
        width:   parseFloat(neomerger_get_value('txt2img_width', 1024)) || 1024,
        height:  parseFloat(neomerger_get_value('txt2img_height', 1024)) || 1024,
        seed:    seed,
        sampler: neomerger_get_dropdown('txt2img_sampling', 'Euler a'),
    };
}

// Writes the settings into a set of hidden fields identified by a prefix.
// prefix 'neomerger_t2i' -> Merge tab ; 'neomerger_pr' -> Block Probe tab.
function neomerger_push_to_prefix(prefix) {
    const s = neomerger_read_t2i();
    neomerger_set_field(prefix + '_prompt',  s.prompt);
    neomerger_set_field(prefix + '_neg',     s.neg);
    neomerger_set_field(prefix + '_steps',   s.steps);
    neomerger_set_field(prefix + '_cfg',     s.cfg);
    neomerger_set_field(prefix + '_w',       s.width);
    neomerger_set_field(prefix + '_h',       s.height);
    neomerger_set_field(prefix + '_seed',    s.seed);
    neomerger_set_field(prefix + '_sampler', s.sampler);
    console.log('[NeoMerger] pushed txt2img -> ' + prefix, s);
}

// Backwards-compatible wrapper for the Merge tab.
function neomerger_push_t2i_to_hidden() {
    neomerger_push_to_prefix('neomerger_t2i');
}

// Generic hook installer: attaches handlers on a button that, just before
// the Gradio click fires, copy txt2img settings into the hidden fields for
// the given prefix. 'pointerdown' covers mouse AND touch; 'keydown' covers
// keyboard activation (Enter/Space on a focused button).
function neomerger_install_hook_for(btn_id, prefix, flag) {
    const btn = gradioApp().querySelector('#' + btn_id);
    if (!btn) return false;
    if (btn.dataset[flag] === '1') return true;
    const push = function () { neomerger_push_to_prefix(prefix); };
    btn.addEventListener('pointerdown', push, true);
    btn.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') push();
    }, true);
    btn.dataset[flag] = '1';
    console.log('[NeoMerger] sync hooks installed on #' + btn_id);
    return true;
}

// Try to install both hooks (Merge & Gen, and Block Probe). Retries until the
// buttons exist in the DOM (tabs are lazily rendered), giving up after ~30 s
// so a mismatched neomerger.py (missing a button) doesn't poll forever.
let neomerger_install_attempts = 0;
function neomerger_try_install() {
    const a = neomerger_install_hook_for('neomerger_mergegen_btn', 'neomerger_t2i', 'neomergerHooked');
    const b = neomerger_install_hook_for('neomerger_probe_btn',    'neomerger_pr',  'neomergerProbeHooked');
    if (a && b) return;
    if (++neomerger_install_attempts >= 60) {
        console.warn('[NeoMerger] gave up installing sync hooks after 60 attempts — '
            + (a ? '' : '#neomerger_mergegen_btn ') + (b ? '' : '#neomerger_probe_btn ')
            + 'not found. Is neomerger.py up to date?');
        return;
    }
    setTimeout(neomerger_try_install, 500);
}

document.addEventListener('DOMContentLoaded', neomerger_try_install);
neomerger_try_install();

// Backwards-compat: keep the old function name in case anything references it.
function neomerger_grab_t2i_settings() {
    neomerger_push_t2i_to_hidden();
    const s = neomerger_read_t2i();
    return [s.prompt, s.neg, s.steps, s.cfg, s.width, s.height, s.seed, s.sampler];
}
