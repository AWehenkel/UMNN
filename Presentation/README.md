# talk-template

A ready-to-fork template for talks, using [remark](https://github.com/gnab/remark), [KaTeX](https://github.com/Khan/KaTeX) and some customised CSS.

## Instructions

- Clone this repository:
```
git clone https://github.com/JoeriHermans/talk-template.git
cd talk-template
```
- Start an HTTP server to serve the slides:
```
python -m http.server 8001
```
- Edit `talk.md` for making your slides.
- Use [decktape](https://github.com/astefanutti/decktape) for exporting your slides to PDF.

## Markup language

Slides are written in Markdown. See the remark [documentation](https://github.com/gnab/remark/wiki/Markdown) for further details regarding the supported features.

This template also comes with grid-like positioning CSS classes (see `assets/grid.css`) and other custom CSS classes (see `assets/style.css`)

## Integration with GitHub pages

Slides can be readily integrated with [GitHub pages](https://pages.github.com/) by hosting the files in a GitHub repositery and enabling Pages in the Settings tab.

## Credits

Template adapted from [Gilles Louppe](https://github.com/glouppe/talk-template).
