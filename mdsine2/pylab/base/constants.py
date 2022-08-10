# Constants
DEFAULT_TAXLEVEL_NAME = 'NA'
SEQUENCE_COLUMN_LABEL = 'sequence'
TAX_IDXS = {'kingdom': 0, 'phylum': 1, 'class': 2,  'order': 3, 'family': 4,
    'genus': 5, 'species': 6, 'asv': 7}
TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'asv']

# Constants
NAME_FORMATTER = '%(name)s'
ID_FORMATTER = '%(id)s'
INDEX_FORMATTER = '%(index)s'
SPECIES_FORMATTER = '%(species)s'
GENUS_FORMATTER = '%(genus)s'
FAMILY_FORMATTER = '%(family)s'
CLASS_FORMATTER = '%(class)s'
ORDER_FORMATTER = '%(order)s'
PHYLUM_FORMATTER = '%(phylum)s'
KINGDOM_FORMATTER = '%(kingdom)s'
PAPER_FORMATTER = '%(paperformat)s'

TAXFORMATTERS = ['%(kingdom)s', '%(phylum)s', '%(class)s', '%(order)s',
    '%(family)s', '%(genus)s', '%(species)s']
TAXANAME_PAPER_FORMAT = float('inf')
