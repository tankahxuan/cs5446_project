from pathlib import Path
import json
import logging
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
logging.getLogger('pypdf').setLevel(logging.ERROR)

source = Path(r'C:\Users\tanka\Dropbox\08-1 NUS Master in Computing AI specialization\CS5546 AI Planning and Decision Making\03 Project')
output = Path(__file__).parent
papers = {
    '01_citylearn': 'CityLearn - Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management.pdf',
    '02_citylearn_v2': 'CityLearn v2 - Energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities.pdf',
    '03_sac': 'Soft Actor-Critic - Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.pdf',
    '04_centralised_sac': 'A Centralised Soft Actor Critic Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn.pdf',
}
manifest = {}
for key, name in papers.items():
    path = source / name
    reader = PdfReader(path)
    pages = [p.extract_text() or '' for p in reader.pages]
    (output / f'{key}.txt').write_text('\n\n'.join(f'PDF PAGE {i + 1}\n{text}' for i, text in enumerate(pages)), encoding='utf-8')
    manifest[key] = {'path': str(path), 'pages': len(pages), 'metadata': str(reader.metadata)}
    print(json.dumps({'paper': key, 'pages': len(pages), 'first_page': pages[0]}, ensure_ascii=False))
(output / 'four_papers_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
