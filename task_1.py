import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import pandas as pd
import csv
import os

# ------------------------
# Конфігурація графа
# ------------------------
edges = [
    # Термінал 1
    ('Термінал 1', 'Склад 1', 25),
    ('Термінал 1', 'Склад 2', 20),
    ('Термінал 1', 'Склад 3', 15),

    # Термінал 2
    ('Термінал 2', 'Склад 2', 10),
    ('Термінал 2', 'Склад 3', 15),
    ('Термінал 2', 'Склад 4', 30),

    # Склади -> Магазини
    ('Склад 1', 'Магазин 1', 15),
    ('Склад 1', 'Магазин 2', 10),
    ('Склад 1', 'Магазин 3', 20),

    ('Склад 2', 'Магазин 4', 15),
    ('Склад 2', 'Магазин 5', 10),
    ('Склад 2', 'Магазин 6', 25),

    ('Склад 3', 'Магазин 7', 20),
    ('Склад 3', 'Магазин 8', 15),
    ('Склад 3', 'Магазин 9', 10),

    ('Склад 4', 'Магазин 10', 20),
    ('Склад 4', 'Магазин 11', 10),
    ('Склад 4', 'Магазин 12', 15),
    ('Склад 4', 'Магазин 13', 5),
    ('Склад 4', 'Магазин 14', 10),
]

terminals = ['Термінал 1', 'Термінал 2']
warehouses = ['Склад 1', 'Склад 2', 'Склад 3', 'Склад 4']
shops = ['Магазин ' + str(i) for i in range(1, 15)]

# Штучні джерело і стік для зведення багатоджерельної задачі до одно-джерельної
SOURCE = 'source'
SINK = 'sink'

# ------------------------
# Побудова матриці/словника пропускних здатностей
# ------------------------
def build_capacity(edges):
    cap = defaultdict(lambda: defaultdict(int))
    for u, v, c in edges:
        cap[u][v] = c
    # source -> terminals: capacity = сума виходів терміналу (щоб термінал не був штучно обмежений)
    for t in terminals:
        cap[SOURCE][t] = sum(cap[t].values())
    # shops -> sink: capacity = сумарна потенційна вхідна місткість магазину
    incoming = defaultdict(int)
    for u, v, c in edges:
        if v.startswith('Магазин'):
            incoming[v] += c
    for s, total_in in incoming.items():
        cap[s][SINK] = total_in
    return cap

# ------------------------
# BFS для знаходження доповнювального шляху (Едмондса-Карпа)
# ------------------------
def bfs_find_path(capacity, flow, s, t):
    parent = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in capacity[u]:
            residual = capacity[u][v] - flow[u][v]
            if v not in parent and residual > 0:
                parent[v] = u
                if v == t:
                    # зібрати шлях
                    path = []
                    cur = t
                    while cur != s:
                        prev = parent[cur]
                        path.append((prev, cur))
                        cur = prev
                    path.reverse()
                    return path
                q.append(v)
    return None

# ------------------------
# Едмондс-Карп: обчислення max-flow і повернення flow-матриці
# ------------------------
def edmonds_karp(capacity, s, t, verbose=True):
    flow = defaultdict(lambda: defaultdict(int))
    max_flow = 0
    iteration = 0
    while True:
        path = bfs_find_path(capacity, flow, s, t)
        if not path:
            break
        iteration += 1
        # мінімальна залишкова здатність на шляху
        residuals = [capacity[u][v] - flow[u][v] for (u, v) in path]
        f = min(residuals)
        # прокачуємо f по всьому шляху
        for (u, v) in path:
            flow[u][v] += f
            flow[v][u] -= f
        max_flow += f
        if verbose:
            print(f"[Ітерація {iteration}] Знайдено шлях:", " -> ".join([path[0][0]] + [v for _, v in path]), f": додано {f}")
    return flow, max_flow

# ------------------------
# Декомпозиція потоку на s->...->t шляхи (для читабельного звіту)
# ------------------------
def decompose_flow(flow, s, t):
    # робимо копію позитивних потоків на оригінальних напрямках
    residual_flow = defaultdict(int)
    for (u, vs) in flow.items():
        for v, f in vs.items():
            # враховуємо тільки оригінальні напрямки (ті, що мали capacity > 0 в початковому графі)
            if f > 0:
                residual_flow[(u, v)] += f

    paths = []
    while True:
        # BFS по ребрах з позитивним потоком
        q = deque([s])
        parent = {s: None}
        while q and t not in parent:
            u = q.popleft()
            for (x, y), val in residual_flow.items():
                if x == u and val > 0 and y not in parent:
                    parent[y] = x
                    q.append(y)
        if t not in parent:
            break
        # збираємо шлях і мінімальний потік
        cur = t
        path = []
        f = float('inf')
        while cur != s:
            prev = parent[cur]
            path.append((prev, cur))
            f = min(f, residual_flow[(prev, cur)])
            cur = prev
        path.reverse()
        # віднімаємо поток від residual_flow
        for (u, v) in path:
            residual_flow[(u, v)] -= f
        paths.append((path, f))
    return paths

# ------------------------
# Збір таблиці Terminal -> Shop
# ------------------------
def aggregate_terminal_shop(paths):
    tm = defaultdict(int)
    for path, f in paths:
        term = None
        shop = None
        # шукати перші вузли, що відповідають форматам
        for (u, v) in path:
            if u.startswith('Термінал'):
                term = u
            if v.startswith('Магазин'):
                shop = v
        if term and shop:
            tm[(term, shop)] += f
    return tm

# ------------------------
# Виведення зрозумілих результатів у консоль
# ------------------------
def pretty_print_results(flow, max_flow, paths, tm_table):
    print("\n" + "="*60)
    print(f"Максимальний сумарний потік у мережі: {max_flow} одиниць")
    print("="*60 + "\n")

    print("Декомпозиція потоку (s -> ... -> t):")
    for i, (p, f) in enumerate(paths, start=1):
        nodes = [p[0][0]] + [v for _, v in p]
        print(f"  {i}. {' -> '.join(nodes)}  :  {f}")
    print()

    print("Таблиця фактичних потоків Термінал -> Магазин (включно з нульовими):")
    # зберемо повну таблицю зі всіма магазинами
    rows = []
    for t in terminals:
        for s in shops:
            rows.append({
                'Термінал': t,
                'Магазин': s,
                'Потік': tm_table.get((t, s), 0)
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()

    # Сумарні потоки по терміналах
    tot_by_term = df.groupby('Термінал')['Потік'].sum()
    print("Сумарні потоки по терміналах:")
    for t, val in tot_by_term.items():
        print(f"  {t}: {val} одиниць")
    print()

    # Потоки по магазинах (агрегація по магазинам)
    tot_by_shop = df.groupby('Магазин')['Потік'].sum()
    min_shops = tot_by_shop[tot_by_shop == tot_by_shop.min()].index.tolist()
    print("Сумарні потоки по магазинах (кількість кожного магазину):")
    # виведемо у зручному форматі
    for shop in shops:
        print(f"  {shop}: {int(tot_by_shop.get(shop, 0))}")
    print()

# ------------------------
# Побудова та візуалізація мережі (networkx)
# ------------------------
def draw_flow_graph(capacity, flow, filename=None):
    G = nx.DiGraph()
    # додаємо всі вершини зі словника capacity
    nodes = set()
    for u in capacity:
        nodes.add(u)
        for v in capacity[u]:
            nodes.add(v)
    for n in nodes:
        G.add_node(n)
    # ребра: capacity і факт. потік
    for u in capacity:
        for v, c in capacity[u].items():
            if c > 0:
                f = flow[u][v] if v in flow[u] else 0
                G.add_edge(u, v, capacity=c, flow=f)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 9))
    # вузли
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # ребра: ширина стрілки = max(0.1, потік) для наочності
    edge_widths = []
    for u, v in G.edges():
        f = G[u][v].get('flow', 0)
        # масштабування: трохи збільшуємо щоб товщина була видимішою
        edge_widths.append(max(0.1, f))

    # кольори: синім позначаємо ті ребра, де f>0, сірим — де 0
    edge_colors = []
    for u, v in G.edges():
        edge_colors.append('blue' if G[u][v].get('flow', 0) > 0 else 'gray')

    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, arrows=True, arrowstyle='-|>', arrowsize=16)
    # підписи ребер потік/ємність
    edge_labels = { (u, v): f"{int(G[u][v].get('flow',0))}/{int(G[u][v].get('capacity',0))}" for u, v in G.edges() }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Фактичні потоки в мережі (потік/ємність)")
    plt.axis('off')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Граф збережено у файл: {filename}")
    plt.show()

# ------------------------
# Збереження таблиці в CSV
# ------------------------
def save_tm_csv(tm_table, out_path='terminal_shop_flows.csv'):
    rows = []
    for t in terminals:
        for s in shops:
            rows.append([t, s, int(tm_table.get((t, s), 0))])
    header = ['Термінал', 'Магазин', 'Потік']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Таблиця Термінал→Магазин збережена у {out_path}")

# ------------------------
# Основний сценарій
# ------------------------
def main():
    print("Побудова графа (оригінального, 20 вершин: 2 термінали + 4 склади + 14 магазинів).")
    capacity = build_capacity(edges)

    print("\nЗапускаємо алгоритм Едмондса—Карпа...")
    flow, max_flow = edmonds_karp(capacity, SOURCE, SINK, verbose=True)

    print("\nГенеруємо декомпозицію потоку для читабельного звіту...")
    # Декомпозиція працює тільки по позитивних потоках в оригінальних напрямках.
    # Для цього беремо потоки тільки у напрямках з capacity>0
    flow_pos = defaultdict(lambda: defaultdict(int))
    for u in flow:
        for v, f in flow[u].items():
            # вважатимемо "фактичним" потоком ті позитивні f на оригінальних ребрах capacity[u][v]>0
            if f > 0 and capacity.get(u, {}).get(v, 0) > 0:
                flow_pos[u][v] = f

    paths = decompose_flow(flow_pos, SOURCE, SINK)

    tm_table = aggregate_terminal_shop(paths)
    pretty_print_results(flow_pos, max_flow, paths, tm_table)

    # Зберігаємо таблицю у CSV
    save_tm_csv(tm_table, out_path='terminal_shop_flows.csv')

    # Візуалізація (збереження PNG)
    out_img = 'flow_network.png'
    draw_flow_graph(capacity, flow_pos, filename=out_img)

    # Визначимо вузькі місця (найменші capacity в оригінальному графі)
    all_edges = []
    for u in capacity:
        for v, c in capacity[u].items():
            # враховуємо тільки оригінальні ребра (без source-> і без ->sink, але якщо хочеш — можна включити)
            if u not in (SOURCE, ) and v not in (SINK, ):
                if c > 0:
                    all_edges.append((u, v, c))
    all_edges_sorted = sorted(all_edges, key=lambda x: x[2])
    print("\nНайменші пропускні здатності (потенційні вузькі місця):")
    for (u, v, c) in all_edges_sorted[:12]:
        print(f"  {u} -> {v} : {c}")

    print("\nГотово. Файли: 'terminal_shop_flows.csv' (таблиця), 'flow_network.png' (граф).")

if __name__ == "__main__":
    main()