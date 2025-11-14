# OCR-D extension to compute table structures

> ocrd_tables is an OCR‑D compliant workspace processor that tries to conduct table structures by combining column as well as text line detection results.
> 
> The row clustering is based on "[ClusTi: Clustering Method for Table Structure Recognition in Scanned Images](https://link.springer.com/article/10.1007/s11036-021-01759-9)".
     
_Disclaimer_: Work in progeress

## Installation

```commandline
pip install .
```     

Or install via Docker:
```
- docker compose build
- docker-compose run ocrd-tables
```
For CPU only:
```
- docker compose build ocrd-tables-cpu
- docker-compose run ocrd-tables-cpu
```    

## Quick Start

```commandline
ocrd-tables \
  -I OCR-D-SEG-COLUMN,OCR-D-SEG-TEXTLINE-TABLE \
  -O OCR-D-TABLE-CELLS \
  -p '{"dbscan_eps":0.6,"dbscan_min_samples":2,"col_pad_frac":0.02}'
```