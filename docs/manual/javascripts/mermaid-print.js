// PDF build only: render Mermaid fences into the normal DOM (no shadow root) so print CSS can size them.
(async function () {
  const blocks = document.querySelectorAll('pre.mermaid-print');
  if (!blocks.length) return;
  mermaid.initialize({ startOnLoad: false, theme: 'neutral', fontFamily: 'Helvetica, Arial, sans-serif', flowchart: { useMaxWidth: true, htmlLabels: false }, timeline: { useMaxWidth: true } });
  let i = 0;
  for (const pre of blocks) {
    const code = pre.textContent;
    try {
      const { svg } = await mermaid.render('mmd' + (i++), code);
      const div = document.createElement('div');
      div.className = 'mermaid-figure';
      div.innerHTML = svg;
      pre.replaceWith(div);
    } catch (e) { console.log('mermaid error', e); }
  }
})();
