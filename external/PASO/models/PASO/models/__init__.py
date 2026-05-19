from models.PASO.models.PASO_GEP import PASO_GEP
from models.PASO.models.PASO_GEP_CNV import PASO_GEP_CNV
from models.PASO.models.PASO_GEP_CNV_MUT import PASO_GEP_CNV_MUT
from models.PASO.models.PASO_TCGA_Classifier_GEP import PASO_GEP_TCGA_Classifier
from models.PASO.models.PASO_GEP_MUT import PASO_GEP_MUT
from models.PASO.models.PASO_V2_GEP import Conv_TransMCA_GEP_V2
from models.PASO.models.PASO_Non_Attention_GEP import PASO_GEP_WithOut_Attention

# More models could follow
MODEL_FACTORY = {
    'PASO_GEP_CNV_MUT': PASO_GEP_CNV_MUT,
    'PASO_GEP_CNV': PASO_GEP_CNV,
    'PASO_GEP': PASO_GEP,
    'PASO_GEP_MUT': PASO_GEP_MUT,
    'PASO_GEP_V2': Conv_TransMCA_GEP_V2,
    'PASO_GEP_NON_ATT': PASO_GEP_WithOut_Attention,
    'PASO_GEP_Classifier': PASO_GEP_TCGA_Classifier
}
