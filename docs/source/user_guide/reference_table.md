# Reference Table

## CD models

| Model   |               DataTPL |    TrainTPL    | EvalTPL                                               |
| :------ | ---------------------: | :-------------: | ------------------------------------------------------ |
| IRT     |         CDInterDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| MIRT    |         CDInterDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| NCDM    | CDInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL |
| CNCD_Q  |           CNCDQDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| CNCD_F  |           CNCDFDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| DINA    | CDInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL |
| HierCDF |         HierCDFDataTPL | GeneralTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL |
| CDGK    |            CDGKDataTPL | GeneralTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL |
| CDMFKC  | CDInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| ECD     |             ECDDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| IRR     |             IRRDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| KaNCD   | CDInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL |
| KSCD    | CDInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| MGCD    |            MGCDDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| RCD     |             RCDDataTPL | GeneralTrainTPL | PredictionEvalTPL                            |
| DCD     |             CCDDataTPL | DCDTrainTPL | PredictionEvalTPL, InterpretabilityEvalTPL       |
| FairCD  |          FAIRDataTPL | AdversarialTrainTPL | PredictionEvalTPL, FairnessEvalTPL       |

## KT models

| Model        |                DataTPL |    TrainTPL    | EvalTPL                    |
| :----------- | ----------------------: | :-------------: | --------------------------- |
| AKT          | KTInterDataTPLCptUnfold | GeneralTrainTPL | PredictionEvalTPL |
| ATKT         | KTInterDataTPLCptUnfold |  AtktTrainTPL   | PredictionEvalTPL |
| CKT          |  KTInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| CL4KT        |            CL4KTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| CT_NCM       | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DeepIRT     |  KTInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DIMKT        |            DIMKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DKT          |          KTInterDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DKTDSC      |           DKTDSCDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DKTForget   |        DKTForgetDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DKT_plus         |          KTInterDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DKVMN        |  KTInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| DTransformer | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| EERNN        |            EERNNDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| EKT          |            EKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| GKT          |  GKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| HawkesKT     | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| IEKT         |  KTInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| KQN          |          KTInterDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| LPKT         |             LPKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| LPKT_S       |             LPKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| QDKT         |             QDKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| QIKT         |  KTInterExtendsQDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| RKT          |              RKTDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| SAINT        | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| SAINT_plus       | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| SAKT         |          KTInterDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| SimpleKT     | KTInterCptUnfoldDataTPL | GeneralTrainTPL | PredictionEvalTPL |
| SKVMN        |          KTInterDataTPL | GeneralTrainTPL | PredictionEvalTPL |
