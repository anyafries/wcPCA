rm(list = ls())
library(FactoMineR)

get_continent <- function(site_ids) {
  # Define a mapping of country codes to continents
  country_to_continent <- list(
    "AR" = "South America",
    "AT" = "Europe",
    "AU" = "Oceania",
    "BE" = "Europe",
    "BR" = "South America",
    "BW" = "Africa",
    "CA" = "North America",
    "CG" = "Africa",
    "CH" = "Europe",
    "CN" = "Asia",
    "CZ" = "Europe",
    "DE" = "Europe",
    "DK" = "Europe",
    "ES" = "Europe",
    "FI" = "Europe",
    "FR" = "Europe",
    "GF" = "South America",
    "GH" = "Africa",
    "HU" = "Europe",
    "IE" = "Europe",
    "IL" = "Asia",
    "IS" = "Europe",
    "IT" = "Europe",
    "JP" = "Asia",
    "MY" = "Asia",
    "NL" = "Europe",
    "NO" = "Europe",
    "PA" = "North America",
    "PT" = "Europe",
    "RU" = "Asia", # Russia = Asia here
    "SD" = "Africa",
    "SE" = "Europe",
    "SN" = "Africa",
    "UK" = "Europe",
    "US" = "North America",
    "VU" = "Oceania",
    "ZA" = "Africa",
    "ZM" = "Africa"
  )
  
  # Extract the first two letters of the SITE_ID and map them to continents
  continents <- sapply(site_ids, function(site_id) {
    country_code <- substr(site_id, 1, 2) # Get the first two letters
    if (country_code %in% names(country_to_continent)) {
      return(country_to_continent[[country_code]])
    } else {
      return(NA) # Return NA if the country code is not in the mapping
    }
  })
  
  return(continents)
}

EFPN <- read.table("data/InputDataMigliavacca2021.csv", header = T, sep = ",")

## -- Selecting EFPs

EFPs_codes_4_PCA <- c("uWUE","ETmax",
                      "GSmax","G1","EF","EFampl",
                      "GPPsat","NEPmax", "Rb","Rbmax","aCUE",
                      "WUEt")

subset <- names(EFPN) %in% EFPs_codes_4_PCA 

## -- Run PCA withouth multiple imputation

EFP.pca <- PCA(scale(EFPN[,subset]), graph = FALSE)
ind <- get_pca_ind(EFP.pca)
EFPN$PC1 <- ind$coord[,1]
EFPN$PC2 <- ind$coord[,2]
EFPN$PC3 <- ind$coord[,3]

EFPN$continent <- get_continent(EFPN$SITE_ID)

table(EFPN$continent)
head(EFPN)

write.csv(EFPN, file="EFPN_with_pcs_and_continents.csv")
