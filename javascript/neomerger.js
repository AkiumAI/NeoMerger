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

function neomerger_push_t2i_to_hidden() {
    const prompt  = neomerger_get_value('txt2img_prompt', '');
    const neg     = neomerger_get_value('txt2img_neg_prompt', '');
    const steps   = parseFloat(neomerger_get_value('txt2img_steps', 28)) || 28;
    const cfg     = parseFloat(neomerger_get_value('txt2img_cfg_scale', 7)) || 7;
    const width   = parseFloat(neomerger_get_value('txt2img_width', 1024)) || 1024;
    const height  = parseFloat(neomerger_get_value('txt2img_height', 1024)) || 1024;
    let seed      = parseFloat(neomerger_get_value('txt2img_seed', -1));
    if (isNaN(seed)) seed = -1;
    const sampler = neomerger_get_dropdown('txt2img_sampling', 'Euler a');

    neomerger_set_field('neomerger_t2i_prompt',  prompt);
    neomerger_set_field('neomerger_t2i_neg',     neg);
    neomerger_set_field('neomerger_t2i_steps',   steps);
    neomerger_set_field('neomerger_t2i_cfg',     cfg);
    neomerger_set_field('neomerger_t2i_w',       width);
    neomerger_set_field('neomerger_t2i_h',       height);
    neomerger_set_field('neomerger_t2i_seed',    seed);
    neomerger_set_field('neomerger_t2i_sampler', sampler);

    console.log('[NeoMerger] grabbed txt2img settings:',
        { prompt, neg, steps, cfg, width, height, seed, sampler });
}

function neomerger_install_hook() {
    const btn = gradioApp().querySelector('#neomerger_mergegen_btn');
    if (!btn) return false;
    if (btn.dataset.neomergerHooked === '1') return true;
    btn.addEventListener('mousedown', neomerger_push_t2i_to_hidden, true);
    btn.dataset.neomergerHooked = '1';
    console.log('[NeoMerger] mousedown hook installed on Merge & Gen button.');
    return true;
}

function neomerger_try_install() {
    if (neomerger_install_hook()) return;
    setTimeout(neomerger_try_install, 500);
}

document.addEventListener('DOMContentLoaded', neomerger_try_install);
neomerger_try_install();

// Backwards-compat: keep the old function name in case anything references it.
function neomerger_grab_t2i_settings() {
    neomerger_push_t2i_to_hidden();
    return [
        neomerger_get_value('txt2img_prompt', ''),
        neomerger_get_value('txt2img_neg_prompt', ''),
        parseFloat(neomerger_get_value('txt2img_steps', 28)) || 28,
        parseFloat(neomerger_get_value('txt2img_cfg_scale', 7)) || 7,
        parseFloat(neomerger_get_value('txt2img_width', 1024)) || 1024,
        parseFloat(neomerger_get_value('txt2img_height', 1024)) || 1024,
        parseFloat(neomerger_get_value('txt2img_seed', -1)) || -1,
        neomerger_get_dropdown('txt2img_sampling', 'Euler a')
    ];
}
