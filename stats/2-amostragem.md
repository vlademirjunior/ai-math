# Métodos de Amostragem

## Amostragem

- De modo geral, ao coletarmos uma amostra, devemos ter certea que ela será selecionada ao acaso, de maneira aleatória. Em outras palavras, a amostra deve garantir que cada membro da população tenha a mesma chance de ser selecionado.
- Vamos estudar 2 tipos de amostragem: probabilísticas e não-probabilísticas.

## Amostragem probabilistica

- Cada elemento da população tem uma certa probabilidade de ser selecionado (em geral todos os elementos tem a mesma probabilidade)
- Visa garantir aleatoriedade (o acaso) na escolha da amostra.
- Permite que inferencias mais justas sejam realizadas em relação á população.

> Vamos estudar 3 tipos: aletória simples, aleatória sistemática e aleatória estratificada (proporcional)

## Amostragem aleatória simples

- Apartir de uma população, cada elemento tem a mesma probabilidade de ser selecionada e essa probabilidade é  (Prob = 1/200), le-se 1 sobre 200. Essa é a base da amostragem aleatória simples.
- Também se aplica a grupos, por exemplo, se eu tenho uma população de 200 e eu escolho um grupo de 3 pessoas dentro, e a probabilidade de escolher esse grupo seja uma probabilidade de valor X, então se eu selecionar um outro grupo de 3 pessoas eu devo ter essa mesma probabilidade.

## Amostragem ateatória sistemática

- A amostra é criada selecionando-se elementos da população em intervalos constantes, após o primeiro elemento ter sido selecionado aleatoriamente.

> Ex.: Uma empresa quer determinar se clientes em potencial aprovariam a embalagem de um novo produto. Ela decide selecionar uma amostra com 10% dos seus clientes usando a metodologia sistemática. Como se daria a seleçào dessa amostra?

10% => 10/100 = 1/10 (le-se 10 sobre 100, ou seja, 1 sobre 10)
=> 1 a cada 10 clientes dessa empresa será selecionado na amostragem.
*E o primeiro cliente deverá ser selecionado aleatóriamente => rand ou random `random(1, 10)` aleatória em 1 e 10 sendo 1 e 10 inculuídos*
=> supondo que retornou 4, então `n=4`.

- Portanto, a empresa irá selecionar o quarto cliente, seguido do décimo quarto, vigésimo quarto, trigésimo quarto, etc.

## Amostragem aleatória estratificada (dividir em grupos)

- Utilizada quando a população está subdividida em grupos, e queremos que os grupos sejam representados de maneira mais justa possível.
- Para isso, devemos selecionar um número `X` de membros de cada subgrupo, proporcional ao tamnho de cada subgrupo.

> Ex.: Em um condomínio há um total de 15 familias compostas por 1 pessoa, 50 compostas por 2 pessoas, 20 compostas por 3 pessoas e 55 compostas por 4 ou mais pessoas. Uma amostra de 30 familias deve ser selecionada. Quantas familias de cada subgrupo devem ser escolhidas para compor essa amostra?

- Vamos precisar fazer isso de maneira estratificado e proporcional ao tamnho de cada grupo.

Número total de famílias no condomínio: 15 + 50 + 20 + 55 = 140 (ou seja, cada subgrupo tem que ser proporcional a essas 140 familias)

=> Número de famílias com 1 pessoa = 15/140 *30 = 3 (3.21428571429), ou seja, vamos selecionar 3 famílias com 1 pessoa para essa amostra.
=> Número de famílias com 2 pessoas = 50/140* 30 = 11, ou seja, vamos selecionar 11 famílias com 2 pessoa para essa amostra.
=> Número de famílias com 3 pessoas = 20/140 *30 = 4, ou seja, vamos selecionar 4 famílias com 3 pessoa para essa amostra.
=> Número de famílias com 4+ pessoas = 55/140* 30 = 12, ou seja, vamos selecionar 12 famílias com 4 ou mais pessoas para essa amostra.

*Para garantir uma justa proporcionalidade, o condomínio deve selecionar 3 famílias com 1 pessoa, 11 famílias com 2 pessoas, 4 famílias com 3 pessoas e 12 famílias com 4 ou mais pessoas.*
