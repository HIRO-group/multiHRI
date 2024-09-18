LAYOUTS_IMAGES_DIR = 'data/layouts'
QUESTIONNAIRES_DIR = 'data/questionnaires'

SCREENSHOTS_DIR = 'data/screenshots'
GIFS_DIR = 'data/gifs'

class AgentType:
    sp = 'Self play'
    n_1_sp_w_cur = 'N-1-SP with curriculum'
    n_1_sp_ran = 'N-1-SP without curriculum'
    n_1_sp_new_cur = 'N-1-SP with new curriculum: L-H/M/L'
    ALL = [sp, n_1_sp_w_cur, n_1_sp_ran, n_1_sp_new_cur]

class TrainedOnLayouts:
    multiple = 'Multiple'
    one = 'One'

class LearnerType:
    originaler = 'Originaler'
    supporter = 'Supporter'
