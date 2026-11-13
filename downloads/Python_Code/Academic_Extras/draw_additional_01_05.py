"""Original scientific figures for the academic edition, using fixed exact data."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

root = Path(__file__).resolve().parents[3]
figures = root / '04_Рисунки_и_схемы'
for kind in ('SVG', 'PNG', 'PDF'):
    (figures / kind).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold'})
ink, teal, gray = '#17444B', '#16818B', '#D9E4E5'


def save(fig, stem):
    for extension in ('svg', 'png', 'pdf'):
        fig.savefig(figures / extension.upper() / (stem+'.'+extension), dpi=300,
                    facecolor='white')
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7, 3.6), layout='constrained')
bottom = np.zeros(2)
components = [('Квадрат смещения', [0., .01], ink, '///'),
              ('Разброс прогноза', [1/6, 1/150], teal, '...'),
              ('Шум наблюдения', [.25, .25], gray, '')]
for label, heights, color, hatch in components:
    ax.bar([0, 1], heights, bottom=bottom, width=.5, color=color,
           edgecolor=ink, linewidth=.7, hatch=hatch, label=label)
    bottom += np.array(heights)
for x, height in enumerate(bottom):
    ax.text(x, height+.014, f'{height:.6f}'.replace('.', ','), ha='center', color=ink)
ax.set_xticks([0, 1], ['Процедура A\nсредний прогноз 2,0', 'Процедура B\nсредний прогноз 1,9'])
ax.set_ylabel('Ожидаемая квадратичная ошибка')
ax.set_ylim(0, .55)
ax.set_xlim(-.6, 1.6)
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.grid(axis='y', color='#D9D9D9', linewidth=.6)
ax.set_axisbelow(True)
ax.set_title('Точное разложение риска в дискретном примере', loc='left', color=ink)
ax.legend(loc='upper right', frameon=False, fontsize=8)
save(fig, 'Fig_A01_01_Bias_Variance')


fig = plt.figure(figsize=(7, 4.8), layout='constrained')
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
matrices = [np.array([[1, 2, 0], [0, 1, 2]]), np.array([[1, 0], [2, 1], [0, 2]])]
for j, (matrix, title) in enumerate(zip(matrices, ['Прямой оператор A', 'Сопряжённый оператор Aᵀ'])):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(matrix, cmap='Blues', vmin=0, vmax=3)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center', color=ink, fontsize=15)
    ax.set_title(title, fontsize=11, color=ink)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
ax = fig.add_subplot(gs[1, :])
values = [1, 1, 2, 1, 2, 1, 2, 1, 1]
ax.bar(np.arange(1, 10), values, color=gray, edgecolor=ink, linewidth=.8, width=.75)
for x, value in enumerate(values, 1):ax.text(x, value+.06, str(value), ha='center', color=ink)
ax.set_xticks(np.arange(1, 10))
ax.set_ylim(0, 2.7)
ax.set_yticks([0, 1, 2])
ax.set_xlabel('Позиция выхода')
ax.set_ylabel('Сумма вкладов')
ax.set_title('Четыре входные единицы, ядро из трёх единиц, шаг 2', fontsize=10, loc='left', color=ink)
save(fig, 'Fig_A03_01_Adjoint_Overlap')
print('PASS: two original figures in SVG, PNG 300 dpi and vector PDF')
