"""Browser checks for the math shown in report and plain-text UI fields."""

import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "static/app.js").read_text()
HELPERS = SOURCE[SOURCE.index("  const esc ="):SOURCE.index("  function fmtDuration")]
REPORT = SOURCE[SOURCE.index("  // Capture math before Markdown"):SOURCE.index("  /* -------------------------------- timeline")]
TIMELINE = SOURCE[SOURCE.index("  function renderTimeline()"):SOURCE.index("  /* -------------------------------- frames")]


class MathRenderingTests(unittest.TestCase):
    def test_timeline_ocr_equations_render_as_math(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content('<!doctype html><div id="timeline"></div>')
                page.add_script_tag(path=str(ROOT / "static/vendor/katex.min.js"))
                state = '''const state = { jobId: 'job', frames: [], timeline: { scenes: [{
                    scene: 0, start_label: '0:00', end_label: '0:10', title: 'Limit', frames: [],
                    screen_text: ['Embedded limits:', 'x -> 0', 'e^x - 1 - x', 'x^2', 'Lt x->0 (e^(x/2) - 1)^2 / x^2']
                }] } }; const icons = { diagram: '' }; const $ = (selector) => document.querySelector(selector);'''
                page.add_script_tag(content=HELPERS + state + TIMELINE)
                page.evaluate('renderTimeline()')
                self.assertEqual(page.locator('.text-chip .katex').count(), 4)
                self.assertEqual(page.locator('.text-chip').first.inner_text(), 'Embedded limits:')
                self.assertEqual(page.locator('.math-error').count(), 0)
            finally:
                browser.close()

    def test_report_repairs_single_line_display_math_and_formats_contents(self):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.set_content('<!doctype html><div id="note"></div><nav id="toc"></nav><button id="copy-note-btn" hidden></button>')
                for name in ("marked.min.js", "purify.min.js", "katex.min.js"):
                    page.add_script_tag(path=str(ROOT / "static/vendor" / name))
                page.add_script_tag(content='const $ = (selector) => document.querySelector(selector);\n' + HELPERS + REPORT)
                markdown = (
                    '## Limit of $\\frac{x^2-1}{x-1}$\n\n'
                    '$$L=\\frac{1}{2}.\n\n'
                    '### [Next step](https://example.com) with $x\\to 0$\n\n'
                    '$$\n\\frac{a}{b}\n$$\n'
                    '\nThe rule uses $$\\frac{f(x)}{g(x)}$$ here and $$x^2$$ later.\n'
                    '\nSource equation: `\\frac{dN}{dt}`.\n'
                    '\n```tex\n\\frac{a}{b}\n```\n'
                )
                page.evaluate('(source) => renderNote(source)', markdown)
                result = page.evaluate('''() => ({
                    noteMath: document.querySelectorAll('#note .katex').length,
                    tocMath: document.querySelectorAll('#toc .katex').length,
                    rawDelimiters: document.querySelector('#note').innerText.includes('$$'),
                    nestedLinks: !!document.querySelector('#toc a a'),
                    errors: document.querySelectorAll('.math-error').length,
                    inlineCode: document.querySelectorAll('#note p code').length,
                    codeBlock: document.querySelectorAll('#note pre code').length,
                    safeText: mathHtml('Value $\\\\frac{1}{2}$ <img src=x onerror=alert(1)>'),
                    ocrText: mathHtml('x=\\\\frac{1}{2}', true),
                    ocrPower: mathHtml('e^x - 1 - x', true),
                    ocrLimit: mathHtml('Lt x->0 (e^(x/2) - 1)^2 / x^2', true),
                    normalizedLimit: ocrMathExpression('Lt x->0 e^(x/2)'),
                    ocrProse: mathHtml('Embedded limits:', true),
                })''')
                self.assertEqual(result["noteMath"], 7)
                self.assertEqual(result["tocMath"], 2)
                self.assertFalse(result["rawDelimiters"])
                self.assertFalse(result["nestedLinks"])
                self.assertEqual(result["errors"], 0)
                self.assertEqual(result["inlineCode"], 0)
                self.assertEqual(result["codeBlock"], 1)
                self.assertIn('class="katex"', result["safeText"])
                self.assertIn('&lt;img', result["safeText"])
                self.assertIn('class="katex"', result["ocrText"])
                self.assertIn('class="katex"', result["ocrPower"])
                self.assertIn('class="katex"', result["ocrLimit"])
                self.assertEqual(result["normalizedLimit"], r'\lim_{x\to 0} e^{x/2}')
                self.assertEqual(result["ocrProse"], 'Embedded limits:')
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
