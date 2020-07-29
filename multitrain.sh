# 2 model types (BNCNN, SVM)

# 10 datasets (all HCP tasks separately + all)

# 12 outcomes ...
# 3 multi-outcome continuous (all OCEAN, all soft-cluster, non-spurious soft-cluster)
# 8 single-outcome continuous (OCEAN separately, each non-spurious soft-clusters {2,6,9})
# 1 multiclass (hardcluster)

datasets=('rest1' 'working_memory' 'gambling' 'motor' 'rest2' 'language' 'social' 'relational' 'faces')
outcomes=('NEOFAC_O' 'NEOFAC_C' 'NEOFAC_E' 'NEOFAC_A' 'NEOFAC_N' 'softcluster_2' 'softcluster_6' 'softcluster_9' 'hardcluster' )
models=('BNCNN' 'SVM')

# multiple tasks, multiple outcomes (only BNCNN)
python dof_parser.py -v -mo BNCNN -cd HCP_alltasks_268 -po softcluster_1 -po softcluster_2 -po softcluster_3 -po softcluster_4 -po softcluster_5 -po softcluster_6 -po softcluster_7 -po softcluster_8 -po softcluster_9 -po softcluster_10 -po softcluster_11 -po softcluster_12 -po softcluster_13 -ct rest1 -ct working_memory -ct gambling -ct motor -ct rest2 -ct language -ct social -ct relational -ct faces
python dof_parser.py -v -mo BNCNN -cd HCP_alltasks_268 -po softcluster_2 -po softcluster_6 -po softcluster_9 -ct rest1 -ct working_memory -ct gambling -ct motor -ct rest2 -ct language -ct social -ct relational -ct faces
python dof_parser.py -v -mo BNCNN -cd HCP_alltasks_268 -po NEOFAC_O -po NEOFAC_C -po NEOFAC_E -po NEOFAC_A -po NEOFAC_N -ct rest1 -ct working_memory -ct gambling -ct motor -ct rest2 -ct language -ct social -ct relational -ct faces

# single task, multiple outcomes (only BNCNN)
for dataset in "${datasets[@]}"; do
  python dof_parser.py -v -ct "$dataset" -mo BNCNN -cd HCP_alltasks_268 -po NEOFAC_O -po NEOFAC_C -po NEOFAC_E -po NEOFAC_A -po NEOFAC_N
  python dof_parser.py -v -ct "$dataset" -mo BNCNN -cd HCP_alltasks_268 -po softcluster_2 -po softcluster_6 -po softcluster_9
  python dof_parser.py -v -ct "$dataset" -mo BNCNN -cd HCP_alltasks_268 -po softcluster_1 -po softcluster_2 -po softcluster_3 -po softcluster_4 -po softcluster_5 -po softcluster_6 -po softcluster_7 -po softcluster_8 -po softcluster_9 -po softcluster_10 -po softcluster_11 -po softcluster_12 -po softcluster_13
done

# multiple tasks, single outcome
for model in "${models[@]}"; do
  for outcome in "${outcomes[@]}"; do
    python dof_parser.py -v -po "$outcome" -mo "$model" -cd HCP_alltasks_268 -ct rest1 -ct working_memory -ct gambling -ct motor -ct rest2 -ct language -ct social -ct relational -ct faces
  done
done

# single task, single outcome
for model in "${models[@]}"; do
  for outcome in "${outcomes[@]}"; do
    for dataset in "${datasets[@]}"; do
      python dof_parser.py -v -ct "$dataset" -po "$outcome" -mo "$model" -cd HCP_alltasks_268
    done
  done
done
