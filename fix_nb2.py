import json
with open('Scripts/plot-notebook.ipynb', 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if 'ax.set_ylabel("F1-score' in line or "plt.ylabel('F1-score" in line or 'plt.ylabel("F1-score' in line:
                new_source.append('    plt.ylim(0, 100)\n')
        cell['source'] = new_source
with open('Scripts/plot-notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
