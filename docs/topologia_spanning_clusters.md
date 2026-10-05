# Topologia: spanning clusters (schema 2)

Os novos arquivos `raw_fractions/.../counts/*_counts.topology` são JSON.
Cada entrada de `samples` corresponde a uma amostra estabilizada, inclusive
quando não existe spanning cluster. Um cluster é spanning se, usando apenas
sítios e ligações dentro da janela, conecta `anchor_z` a `anchor_z + L - 1`.
As direções laterais continuam periódicas; a direção de crescimento é aberta.
Em bond percolation, somente ligações abertas conectam os sítios.

- `sample_index`: índice da amostra, começando em zero.
- `anchor_z`, `t_stab`: posição da janela e instante da coleta.
- `num_spanning_clusters`: quantidade de clusters atravessantes distintos.
- `largest_spanning_cluster_index`: `0`, ou `null` quando não há nenhum.
- `spanning_clusters`: clusters ordenados por número de sítios decrescente;
  empates são resolvidos pelo menor `component_seed`.

Cada cluster contém `cluster_index`, `component.num_sites`,
`component.num_bonds`, `d_bulk`, `d_min` e todas as contagens brutas:
`box_counting_component`, `minimum_path_yardstick` e `chemical_distance`.
O índice 0 é sempre o maior spanning cluster, mesmo que exista um componente
não atravessante maior. Os demais índices fornecem os mesmos observáveis.
Não são calculadas nem gravadas contagens para `d_hull` ou `d_hull,ext`.

## Estimativas e dados brutos

`d_bulk.value` é a inclinação OLS de `log(mean(N_boxes))` contra
`log(L/epsilon)`, fazendo primeiro a média sobre os offsets de cada escala.
`d_min.value` usa `log(N_spheres)` contra `log(L/R)` para o yardstick do
menor caminho base–topo daquele cluster. São estimativas por amostra e cluster;
o intervalo automático não substitui a escolha de uma região de escala na
análise científica. A análise radial de distância química permanece disponível
em `chemical_distance`, inclusive para uma estimativa alternativa de `d_min`.

Ambos os ajustes usam as escalas existentes entre 2 e L/4, com contagem maior
que 1 e pelo menos três escalas. Cada objeto informa `value`, `r_squared`,
`num_scales`, `min_scale`, `max_scale` e `reason_if_undefined`.
Quando faltam escalas, `value` e `r_squared` são `null`; as contagens continuam
salvas. `meta.dimension_fit` registra a convenção.

```python
import json

with open("caminho/arquivo_counts.topology") as f:
    dados = json.load(f)

for amostra in dados["samples"]:
    print(amostra["sample_index"], amostra["num_spanning_clusters"])
    for cluster in amostra["spanning_clusters"]:
        print(cluster["cluster_index"], cluster["component"]["num_sites"],
              cluster["d_bulk"]["value"], cluster["d_min"]["value"])
    maior = (amostra["spanning_clusters"][0]
             if amostra["num_spanning_clusters"] else None)
```

A coleta mantém os critérios anteriores de ativação (`SOP_FRACTAL_FORCE_COUNTS=1`
para outros tamanhos; o gerador `python/run_samples_topological.py` já ativa isso).
Os arquivos antigos não são reescritos. Leitores do schema anterior precisam
passar a percorrer `samples[*].spanning_clusters[*]`; `largest_component` foi
substituído por `component` dentro de cada cluster. Os observáveis legados do
arquivo separado de frações, incluindo geometrias de buracos e perímetros,
continuam com a convenção anterior; eles não são dimensões fractais de hull.

## Validação

```sh
cmake --build build -j 4
c++ -std=c++17 -O0 -fopenmp -Isrc tests/test_spanning_topology.cpp build/libsop_core.a -o /tmp/test_spanning_topology
/tmp/test_spanning_topology /tmp/spanning_topology.json
```

O teste cobre 2D/3D, node/bond, dois spanning clusters com massas diferentes,
um componente não spanning maior, ausência de spanning após cortar os caminhos,
e concordância das contagens entre as representações 2D densa e esparsa.
