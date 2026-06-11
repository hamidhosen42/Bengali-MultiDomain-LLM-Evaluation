import json
with open('Scripts/plot-notebook.ipynb', 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'annot_kws' in line and 'color' in line:
                cell['source'][i] = line.replace(', "color": "white"', '')

with open('Scripts/plot-notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
