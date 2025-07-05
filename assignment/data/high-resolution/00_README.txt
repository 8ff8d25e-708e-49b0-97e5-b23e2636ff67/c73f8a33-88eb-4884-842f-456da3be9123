
This README file was generated on [2023-05-04] by [THOMAS UPTON].
Last updated: [2023-05-22] by Ida Sletten.


-------------------
GENERAL INFORMATION
-------------------
// Title of Dataset: Replication Data for: High resolution daily profiles of tissue adrenal steroids by portable automated collection
// DOI: https://doi.org/10.18710/5TW8YF
// Contact Information
     // Name: Thomas Upton
     // Institution: University of Bristol
     // Email: thomas.upton@bristol.ac.uk
     // ORCID: https://orcid.org/0000-0003-0138-3982

// Contributors: See metadata field Contributor.
// Kind of data: See metadata field Kind of Data.
// Date of data collection/generation: See metadata field Date of Collection.
// Geographic location: See metadata section Geographic Coverage.
// Funding sources: See metadata section Grant Information.

// Description of dataset: 
1. Time series measurements of adrenal hormones, measured in microdialysis fluid 
2. Time series measurements of adrenal hormones and ACTH, measured in blood plasma 
3. Supporting metadata 
4. Non-validated questionnaire responses

Rhythms are intrinsic to endocrine systems, and disruption of these hormone oscillations occurs at very early stages of disease. Since adrenal hormones are secreted with both circadian and ultradian periods, conventional single time point measures provide limited information about rhythmicity, and crucially do not provide information during sleep when many hormones fluctuate from nadir to peak concentrations. If blood sampling is attempted overnight, this necessitates admission to a clinical research unit, can be stressful and disturbs sleep. To overcome this problem, and to measure free hormones within their target tissues, we used microdialysis, an ambulatory fraction collector and liquid-chromatography tandem mass-spectrometry (LC-MS/MS) to obtain high-resolution profiles of tissue adrenal steroids over 24 hours in 214 healthy volunteers. For validation, we compared tissue against plasma measurements in a further seven healthy volunteers. Sample collection from subcutaneous tissue was safe, well tolerated and allowed most normal activities to continue. In addition to cortisol, we identified daily and ultradian variation in free cortisone, corticosterone (CCS), 18-hydroxycortisol (18-OHF), aldosterone, tetrahydrocortisol (THF), allotetrahydrocortisol (aTHF), and the presence of dehydroepiandrosterone sulphate (DHEA-S). We used mathematical and computational methods to quantify the interindividual variability of hormones at different times of the day and develop “dynamic biomarkers” of normality in healthy individuals stratified by sex, age and body mass index (BMI). Our results provide, for the first time, an insight into the dynamics of adrenal steroids in tissue in real world settings and will serve as a normative reference for novel and improved biomarkers of endocrine disorders. (ULTRADIAN, NCT02934399).
--------------------------
METHODOLOGICAL INFORMATION
--------------------------


// Description of sources and methods used for collection/generation of data: 
Data presented here includes results from the healthy volunteer cohort of the Dynamic Hormone Diagnostics (ULTRADIAN) study (https://cordis.europa.eu/project/id/633515; https://clinicaltrials.gov/ct2/show/NCT02934399). 
Briefly, the dataset contains time series measurements of multiple adrenal hormones in tissue interstitial fluid and plasma samples of healthy volunteers, and associated metadata. 

// Methods for processing the data: 
Raw results from liquid-chromatography mass-spectrometry (LC-MS) analysis of microdialysis samples was aligned with sampling metadata using Python code to create the time series presented here. 
Only fields containing data relevant to the current publication are deposited
Data has been deidentified.
The Python code used for data analysis and production is available on request from the authors.

// Facility-, instrument- or software-specific information needed to interpret the data: 
We used  Python (ver. 3.8.12) with numeric and data analysis libraries NumPy, SciPy, Pandas, CosinorPy and plotting libraries Matplotlib and Seaborn.
 

// Standards and calibration information: 
Full details of the standards and calibration of the LC-MS/MS analysis platform are available as part of the Supplementary Methods associated with the main publication.



--------------------
DATA & FILE OVERVIEW
--------------------
// File List: 
00_README.text

\Data files:
	form_912_controls.csv	Key: MasterId. Inclusion form that codes MasterId, gender, smoking status, and height
	form_914_controls.csv	Key: MasterId. Sampling form that codes additional data related to the specific sampling session
	redcap_controls.csv	Key: MasterID. Metadata related to activity during the sampling session, and results from the Device Satisfaction Questionnaire
	md_data_controls.csv	Key: MasterId. Time series hormone data for each participant
	HABS x.csv		Data from n = 7 participants who completed the comparative plasma vs. microdialysis study. Column names with prefix 'm' refer to microdialysis measurements

\Data dictionaries
	DynamicHormoneDiagnostics_DataDictionary_2022-03-02.csv	Dictionary to interpret field data in the file 'redcap_controls.csv'
	ultradian-912.csv					Dictionary to interpret field data in the file 'form_912_controls.csv' 
	ultradian-914.csv					Dictionary to interpret filed data in the file 'form_914_controls.csv'

// Relationship between files, if important: 
Data files (with the exception of the HABS x.csv) are linked using the anonymous MasterId key


// Is this a new version of a previously published dataset? NO



-----------------------------------------
DATA-SPECIFIC INFORMATION FOR CSV files
-----------------------------------------
For files in the \Data files folder:

Data in comma separated value format.
The delimiter is 'comma ,' for all files except for the data dictionaries for Form 912 and 914 which the delimiter is 'semicolon ;'.
Encoding is UTF-8.

// Variable/Column List: 
Refer to the Data dictionary files for description of column names. 

// Missing data codes: 
Blank cell, except in files 'HABS x.csv' where NA is used.

// Specialized formats or other abbreviations used: 
Refer to main text of publication for description of abbreviations.



--------------------------
SHARING/ACCESS INFORMATION
--------------------------

// Links to publications that cite or use the data: See metadata field Related Publication.
// Links/relationships to related data sets: See metadata field Related Datasets.
// Data sources: See metadata field Data Sources.
// Recommended citation: See citation generated by repository.


<
Acknowledgements. This README file template is adapted from the following documents/resources:

AJPS. ‘American Journal of Political Science Qualitative Data Verification Checklist’. Wiley, 11 March 2016. https://ajps.org/wp-content/uploads/2019/01/ajps-qualdata-checklist-ver-1-0.pdf.

Berkeley Library, University of California. How to Write a Good Documentation. Available at: http://guides.lib.berkeley.edu/how-to-write-good-documentation

Cornell University. Guide to writing "readme" style metadata. Available at: https://data.research.cornell.edu/content/readme#bestpractices

Corti, Louise, Veerle Van den Eynden, Libby Bishop, Matthew Woollard, Maureen Haaker, and Scott Summers. Managing and Sharing Research Data: A Guide to Good Practice. 2nd edition. Los Angeles: SAGE, 2019.

Dryad. Best practices for creating reusable data publications. Available at: https://datadryad.org/stash/best_practices#describe

PurpleBooth. A template to make good README.md. Available at: https://gist.github.com/PurpleBooth/109311bb0361f32d87a2

University of Bath. Working with data: Data Documentation and Metadata. Available at: https://library.bath.ac.uk/research-data/working-with-data/data-documentation-metadata

Zalando. Zalando's README Template. Available at: https://github.com/zalando/zalando-howto-open-source/blob/master/READMEtemplate.md#readme
>
