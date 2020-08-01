import pandas as pd  
LabeID_docID = pd.read_csv('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/AnnotationResults/testData/LabeID_docID.csv')
DocumentID = pd.read_csv('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/AnnotationResults/testData/DocumentID.csv')
Label = pd.read_csv('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/AnnotationResults/testData/Label.csv')
project = pd.read_csv('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/AnnotationResults/testData/project.csv')


Doc_lab_match = pd.merge(DocumentID,LabeID_docID,on="document_id")

Doc_lab_proj = pd.merge(Doc_lab_match,project,on="project_id")


Doc_lab_proj_final = pd.merge(Doc_lab_proj,Label,on="label_id")

Doc_lab_proj_final_T= Doc_lab_proj_final.drop_duplicates(subset=['document_id'])

Doc_lab_proj_final_T.to_csv('/home/iialab/PycharmProjects/ArgumentMining/reu.unt.edu/data/AnnotationResults/testData/Doc_lab_proj_final2.csv',index=False)

Doc_lab_proj_final_T = Doc_lab_proj_final_T[ ['document_id','text_x','project_id_x',
                                               'id','label_id','user_id','name','polymorphic_ctype_id','text_y','project_id_y'] ]

Doc_lab_proj_final_T["text_y"] = Doc_lab_proj_final_T["text_y"].astype('category')
Doc_lab_proj_final_T.dtypes

Doc_lab_proj_final_T["text_y"] = Doc_lab_proj_final_T["text_y"].cat.codes
Doc_lab_proj_final_T.head()

Doc_lab_proj_final_T=Doc_lab_proj_final_T.sort_values(by=['text_x'])

grouped = Doc_lab_proj_final_T.groupby(Doc_lab_proj_final_T.project_id_y)

Team3_tiffany = grouped.get_group(10)
Team3_william = grouped.get_group(17)
Team3_pinna = grouped.get_group(18)


from sklearn.metrics import cohen_kappa_score

k_tiffany_william = cohen_kappa_score(Team3_tiffany.text_y , Team3_william.text_y)
k_william_pinna = cohen_kappa_score(Team3_william.text_y, Team3_pinna.text_y)
K_pinna_tiffany = cohen_kappa_score(Team3_tiffany.text_y , Team3_pinna.text_y)

Team2_vincent  = grouped.get_group(16)
Team2_richard  = grouped.get_group(15)
Team2_sanjeev  = grouped.get_group(13)

k_vincent_rich  = cohen_kappa_score(Team2_vincent.text_y , Team2_richard.text_y)
k_rich_sanjeev = cohen_kappa_score(Team2_richard.text_y , Team2_sanjeev.text_y)
k_sanjeev_vincent  = cohen_kappa_score(Team2_sanjeev.text_y , Team2_vincent.text_y)

Team1_lavania  = grouped.get_group(12)
Team1_max  = grouped.get_group(11)

k_max_lavania  = cohen_kappa_score(Team1_lavania.text_y , Team1_max.text_y)

print(k_max_lavania)



