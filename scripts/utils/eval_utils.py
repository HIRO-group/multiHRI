class Eval:
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    ALL = [LOW, MEDIUM, HIGH]

POPULATION_EVAL_AGENTS = {
    'selected_2_chefs_coordination_ring': 'agent_models/Result/Eval/2/pop_selected_2_chefs_coordination_ring',
    'selected_2_chefs_counter_circuit':   'agent_models/Result/Eval/2/pop_selected_2_chefs_counter_circuit',
    'selected_2_chefs_cramped_room':      'agent_models/Result/Eval/2/pop_selected_2_chefs_cramped_room',
}


def print_selected_agents_for_evaluation(selected_agents_for_evaluation):
    for primary_agent, layouts in selected_agents_for_evaluation.items():
        print(f"\nPrimary Agent: {primary_agent.name}")
        for layout_name, unseen_counts in layouts.items():
            print(f"  Layout: {layout_name}")
            for unseen_count, levels in unseen_counts.items():
                print(f"    Unseen Count: {unseen_count}")
                for teammate_lvl, teams in levels.items():
                    print(f"      Teammate Level: {teammate_lvl}")
                    for i, team in enumerate(teams):
                        team_members = [(member.name, member.layout_scores[layout_name]) for member in team]
                        print(f"        Team {i + 1}: {team_members}")


TWO_PLAYERS_LOW_EVAL = [
    'agent_models/StaticADV/2/SP_hd64_seed14/ck_0',
    'agent_models/StaticADV/2/SP_hd64_seed0/ck_0',
    'agent_models/StaticADV/2/SP_hd256_seed13/ck_0',
    'agent_models/StaticADV/2/SP_hd256_seed68/ck_0',
]

TWO_PLAYERS_MEDIUM_EVAL = [
    'agent_models/DummyADV_correct_collision_checker/2/SP_hd64_seed14/ck_1_rew_88.5',
    'agent_models/DummyADV_correct_collision_checker/2/SP_hd64_seed0/ck_1_rew_92.5',
    'agent_models/DummyADV_correct_collision_checker/2/SP_hd256_seed13/ck_1_rew_113.0',
    'agent_models/DummyADV_correct_collision_checker/2/SP_hd256_seed68/ck_1_rew_119.0',
]

TWO_PLAYERS_HIGH_EVAL = [
    'agent_models/StaticADV/2/SP_hd256_seed13/ck_1_rew_252.0',
    'agent_models/StaticADV/2/SP_hd64_seed0/best',
    'agent_models/StaticADV/2/SP_hd256_seed13/best',
    'agent_models/StaticADV/2/SP_hd256_seed68/best',
]

THREE_PLAYERS_LOW_EVAL = [
    'agent_models/Result/Eval/3/SP_hd64_seed11/ck_0',
    'agent_models/Result/Eval/3/SP_hd64_seed11/ck_1_rew_21.77777777777778',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_1_rew_21.333333333333332',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_2_rew_57.77777777777778',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_0',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_1_rew_33.77777777777778',
    'agent_models/Result/Eval/3/SP_hd256_seed42/ck_1_rew_24.0',
    'agent_models/Result/Eval/3/SP_hd256_seed42/ck_2_rew_88.88888888888889'
]

THREE_PLAYERS_MEDIUM_EVAL = [
    'agent_models/Result/Eval/3/SP_hd256_seed42/ck_3_rew_165.33333333333334',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_2_rew_113.33333333333333',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_3_rew_166.22222222222223',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_3_rew_111.55555555555556',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_4_rew_184.88888888888889',
    'agent_models/Result/Eval/3/SP_hd64_seed11/ck_3_rew_113.77777777777777'
]

THREE_PLAYERS_HIGH_EVAL = [
    'agent_models/Result/Eval/3/SP_hd64_seed11/ck_24_rew_338.6666666666667',
    'agent_models/Result/Eval/3/SP_hd64_seed11/ck_24_rew_338.6666666666667',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_35_rew_382.6666666666667',
    'agent_models/Result/Eval/3/SP_hd64_seed1995/ck_16_rew_348.0',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_20_rew_337.3333333333333',
    'agent_models/Result/Eval/3/SP_hd256_seed7/ck_24_rew_324.8888888888889',
    'agent_models/Result/Eval/3/SP_hd256_seed42/ck_18_rew_332.8888888888889',
    'agent_models/Result/Eval/3/SP_hd256_seed42/ck_19_rew_327.55555555555554'
]

FIVE_PLAYERS_HIGH_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL = [
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_55_rew_187.46666666666667',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_54_rew_176.26666666666668',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_56_rew_173.33333333333334',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_53_rew_167.46666666666667',

    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_54_rew_259.46666666666664',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_55_rew_253.86666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_56_rew_257.8666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_57_rew_245.06666666666666',

    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_54_rew_262.6666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_55_rew_256.26666666666665',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_56_rew_253.6',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_57_rew_239.2',

    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_54_rew_234.66666666666666',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_55_rew_237.86666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_56_rew_230.93333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_60_rew_225.06666666666666'
]
# 0 - 200 -> 0 - 60, 60-120, 120-200
FIVE_PLAYERS_MEDIUM_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL = [
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_7_rew_61.6',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_8_rew_65.86666666666666',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_9_rew_66.13333333333334',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_10_rew_84.0',

    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_4_rew_63.733333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_7_rew_84.8',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_8_rew_88.26666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_9_rew_91.46666666666667',

    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_4_rew_62.93333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_5_rew_89.6',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_6_rew_98.93333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_7_rew_109.06666666666666',

    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_6_rew_67.73333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_7_rew_86.4',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_8_rew_105.86666666666666',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_9_rew_128.0'
]

FIVE_PLAYERS_LOW_EVAL = [
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_0',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_1_rew_2.6666666666666665',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_2_rew_6.133333333333334',
    'agent_models/Result/Eval/5/SP_hd64_seed2907/ck_3_rew_19.733333333333334',

    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_0',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_1_rew_1.6',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_2_rew_11.733333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_3_rew_37.86666666666667',

    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_0',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_1_rew_5.333333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_2_rew_16.8',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_3_rew_36.8',

    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_0',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_1_rew_4.266666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_2_rew_13.333333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_3_rew_25.333333333333332'
]


FIVE_PLAYERS_MEDIUM_STORAGE_EVAL = [
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_10_rew_92.53333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_11_rew_95.73333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_12_rew_99.2',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_13_rew_89.86666666666666',

    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_4_rew_63.733333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_7_rew_84.8',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_8_rew_88.26666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_9_rew_91.46666666666667',

    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_4_rew_62.93333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_5_rew_89.6',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_6_rew_98.93333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_7_rew_109.06666666666666',

    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_6_rew_67.73333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_7_rew_86.4',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_8_rew_105.86666666666666',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_9_rew_128.0'
]

FIVE_PLAYERS_HIGH_STORAGE_EVAL = [
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_52_rew_208.26666666666668',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_53_rew_242.13333333333333',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_58_rew_228.26666666666668',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_57_rew_216.26666666666668',

    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_54_rew_259.46666666666664',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_55_rew_253.86666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_56_rew_257.8666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed105/ck_57_rew_245.06666666666666',

    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_54_rew_262.6666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_55_rew_256.26666666666665',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_56_rew_253.6',
    'agent_models/Result/Eval/5/SP_hd256_seed128/ck_57_rew_239.2',

    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_54_rew_234.66666666666666',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_55_rew_237.86666666666667',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_56_rew_230.93333333333334',
    'agent_models/Result/Eval/5/SP_hd256_seed2907/ck_60_rew_225.06666666666666'
]