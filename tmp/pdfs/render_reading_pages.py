from pathlib import Path
import json
import subprocess
from PIL import Image, ImageOps

root = Path(__file__).parent
papers = json.loads((root / 'four_papers_manifest.json').read_text(encoding='utf-8'))
poppler = Path(r'C:\Users\tanka\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
for paper, pages in {'01_citylearn': [7], '02_citylearn_v2': [8, 11], '03_sac': [5, 6], '04_centralised_sac': [3, 4]}.items():
    images = []
    for page in pages:
        prefix = root / f'{paper}_page_{page}'
        subprocess.run([str(poppler), '-f', str(page), '-l', str(page), '-singlefile', '-scale-to', '1500', '-png', papers[paper]['path'], str(prefix)], check=True, capture_output=True)
        images.append(Image.open(prefix.with_suffix('.png')).convert('RGB'))
    width = sum(i.width for i in images)
    canvas = Image.new('RGB', (width, max(i.height for i in images)), 'white')
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    target = root / f'{paper}_reading_pages.png'
    canvas.save(target)
    print(target)
