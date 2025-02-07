import potentials

potdb = potentials.Database(local=True, remote=True)
entries, entries_df = potdb.get_lammps_potentials(pot_dir_style='id', verbose=True, return_df=True)

results = {}
potdb.widget_lammps_potential(lammps_potentials=entries, lammps_potentials_df=entries_df,results=results)

lmppot = results['lammps_potential']
print(lmppot.id)
print('All symbols used by the potential:')
print(lmppot.symbols)

print(lmppot.pair_info())

print(lmppot.pair_info(symbols=lmppot.symbols[0], comments=False))



