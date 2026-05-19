# Set locale to UTF-8
Sys.setlocale("LC_ALL", "en_US.UTF-8")
setwd("Please enter your destination folder")
library("rjson")
library(stringr)
json <- jsonlite::fromJSON("metadata.cart.2024-11-20.json")
file_sample0 <- json[c('file_name','associated_entities')]
# View(file_sample0)
file_sample0$sample_id <- sapply(file_sample0$associated_entities,function(x){x[,1]})
file_sample <- subset(file_sample0, select = -associated_entities)

file_sample$file_name <- sapply(strsplit(file_sample$file_name,split='e_counts.tsv'),function(x){x[1]})  #对file_name列进行切割，以和表达矩阵的文件名一致
# View(file_sample)
count_file <- list.files('gep_file',pattern = '*rna_seq.augmented_star_gen',recursive = TRUE)
count_file_name <- strsplit(count_file,split='/')
count_file_name <- sapply(count_file_name,function(x){x[2]})

count_file_name <- sapply(strsplit(count_file_name,split="e_counts.tsv"),function(x){x[1]})
RNAseq_Ensembl_matrix <- data.frame()
# clinical_drug_merged_simplified.csv
clinical_drug_merged_simplified <- read.csv('clinical_drug_merged_simplified.csv')
# bcr_patient_barcode
bcr_patient_barcode <- as.character(clinical_drug_merged_simplified$bcr_patient_barcode)

for (i in seq_along(count_file_name)){
  path <- paste0('gep_file//',count_file[i])
  # Skip if path contains "parcel"
  if (grepl("parcel", path)) {
    next
  }
  curr_file_bcr_code <- str_extract(file_sample[which(file_sample$file_name == count_file_name[i]),'sample_id'], "^([^-]+-[^-]+-[^-]+)")[1]
  # Skip if curr_file_bcr_code is not in bcr_patient_barcode
  # print(curr_file_bcr_code)
  if (!(curr_file_bcr_code %in% bcr_patient_barcode)) {
    next
  }
  data0 <- read.table(path,fill = TRUE,header = TRUE)
  # Filter gene_type for protein_coding
  data0 <- data0[data0$gene_type == 'protein_coding',]
  # Extract the unstranded columns to get the COUNT matrix.
  # If you want to extract fpkm-unstranded change to data0[-c(1:4),c(1,8)]
  # and fpkm-up-unstranded change to data0[-c(1:4),c(1,9)]
  data <-data0[-c(1:4),c(1,7)] # tpm ;
  colnames(data)[2] <- curr_file_bcr_code

  RNAseq_Ensembl_matrix <- if (nrow(RNAseq_Ensembl_matrix) == 0) data else merge(RNAseq_Ensembl_matrix, data, by = "gene_id")
}

Ensembl_Symbol <- data0[c('gene_id','gene_name')]
RNAseq_Ensembl_matrix <- merge(Ensembl_Symbol, RNAseq_Ensembl_matrix)
RNAseq_Ensembl_matrix <- RNAseq_Ensembl_matrix[-1]

# gene_name column has duplicates, need to delete duplicate genes, here to keep each gene maximum expression result
RNAseq_Ensembl_matrix <- aggregate(. ~ gene_name, data=RNAseq_Ensembl_matrix, max)


# Log2(TPM+1)
RNAseq_Ensembl_matrix[, 2:ncol(RNAseq_Ensembl_matrix)] <- log2(RNAseq_Ensembl_matrix[, 2:ncol(RNAseq_Ensembl_matrix)]+1)

write.csv(RNAseq_Ensembl_matrix,'TCGA_RNAseq_Log_TPM_Ensembl_matrix.csv',row.names = FALSE)

print("RNAseq_TPM_Ensembl_matrix.csv has been generated successfully!")
