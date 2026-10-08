FROM informaticsmatters/rdkit-cartridge-debian@sha256:99d870945bc30ebbea2661a4c8644f67a0f84d4370f9ffd407370daac24d3e86

LABEL org.opencontainers.image.source="https://github.com/lamalab-org/corral"

COPY reactions_raw_db.sql.gz /seed/
COPY reactions_production_db.sql.gz /seed/
COPY restore.sh /docker-entrypoint-initdb.d/01-restore.sh
RUN chmod +x /docker-entrypoint-initdb.d/01-restore.sh
