FROM informaticsmatters/rdkit-cartridge-debian:latest

COPY reactions_raw_db.sql.gz /seed/
COPY reactions_production_db.sql.gz /seed/
COPY 01-restore.sh /docker-entrypoint-initdb.d/01-restore.sh
RUN chmod +x /docker-entrypoint-initdb.d/01-restore.sh
