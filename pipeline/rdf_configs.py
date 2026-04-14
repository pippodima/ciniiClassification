# ============================================================================
# CONSTANTS
# ============================================================================

# XML namespaces for RDF parsing
NAMESPACES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
    "jpcoar": "https://github.com/JPCOAR/schema/blob/master/2.0/",
    # this is the *default* namespace in your RDF files
    "cir": "https://cir.nii.ac.jp/schema/1.0/",
}
# CSV field names
CSV_FIELDS = ['file', 'title', 'abstract']
