# Description: Download clinical data from TCGA using TCGAbiolinks package
library(TCGAbiolinks)

projects <- TCGAbiolinks::getGDCprojects()$project_id ## Get all TCGA projects
projects <- projects[grepl('^TCGA', projects, perl=TRUE)]

projects

sapply(projects, function(project){

  query <- GDCquery(project = project,
                    data.category = "Clinical",
                    file.type = "xml")
  GDCdownload(query)
  clinical <- GDCprepare_clinic(query, clinical.info = "patient")
  saveRDS(clinical,file = paste0(project,"_clinical.rds"))
})