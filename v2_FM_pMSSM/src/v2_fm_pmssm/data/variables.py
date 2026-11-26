from .dataset import Variable

TIME_ID = 0  # id reserved for the time token

input_params = ['IN_M_1', 'IN_M_2', 'IN_M_3', 'IN_At', 'IN_Ab', 'IN_Atau', 'IN_mu', 
                'IN_tanb', 'IN_mA',  'IN_mqL3', 'IN_mtR', 'IN_mbR']

neutralino_masses = ['SP_m_chi_10', 'SP_m_chi_20', 'SP_m_chi_30', 'SP_m_chi_40']
chargino_masses = ['SP_m_chi_1p', 'SP_m_chi_2p']
stop_masses = ['SP_m_t_1', 'SP_m_t_2']
neutralino_2_branching_fractions = ['SP_BF_chi_20_to_chi_10_Z', 'SP_BF_chi_20_to_chi_10_h', 'SP_BF_chi_20_to_chi_10',
                                    'SP_BF_chi_20_to_chi_1p_W', 'SP_BF_chi_20_to_chi_1p']

# Ids are fixed: never insert into the lists above; append new variables with the next free id (currently 28)
all_var  = [Variable(p, 1 + i, "IN", "theory_parameter", "standard") for i, p in enumerate(input_params)]  # ids 1-12
all_var += [Variable("MO_Omega", 13, "MO", "dark_matter", "log10", invalid=(-1,))]
all_var += [Variable("SP_m_h", 14, "SP", "spheno_higgs_mass", "standard", invalid=(-1,))]  # light Higgs (125 GeV)
all_var += [Variable(p, 15 + i, "SP", "spheno_electroweakino_mass", "standard", invalid=(-1,)) for i, p in enumerate(neutralino_masses)]  # ids 15-18
all_var += [Variable(p, 19 + i, "SP", "spheno_electroweakino_mass", "standard", invalid=(-1,)) for i, p in enumerate(chargino_masses)]    # ids 19-20
all_var += [Variable(p, 21 + i, "SP", "spheno_squark_mass", "standard", invalid=(-1,)) for i, p in enumerate(stop_masses)]    # ids 21-22
all_var += [Variable(p, 23 + i, "SP", "spheno_electroweakino_branching_fraction", "logit", invalid=(-1,)) for i, p in enumerate(neutralino_2_branching_fractions)]    # ids 23-27

_ids = [v.id for v in all_var]
assert len(set(_ids)) == len(_ids) and TIME_ID not in _ids, "variable ids must be unique and not TIME_ID"