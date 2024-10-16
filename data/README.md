We synthesize a dataset extending BirdSQL (https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird), which has DBs spanning over diverse domains including medical, finance, education, sports, and games. The queries posed on those DBs require domain understanding. We take 11 Dev split DBs into consideration, each belonging to a different domain, with a total of 1534 parallel NL-SQL query pairs.

The dataset provides question specific domain knowledge (evidences), in the form of NL domain statements. Such query specific statements are not available for real world DBs, so to simulate real-world enterprise settings, we form 50-50% random splits of the queries in each database of the Dev split, referred to as the IN and OUT splits. The domain statements for all queries in the IN split are combined to form the domain repository for retrieval while the domain statements for the OUT set are not included.

We find that most 77% queries in the OUT sets of all the DBs have no domain statement overlap with the IN set queries for all DBS. Whereas, remaining queries in the OUT set have only partial oracle domain statement overlap with the queries in the IN set. This allows us to simulate the realistic scenario of domain repository construction without knowledge of the queries that may get posed on the DB. More importantly, we do not assume the availability of the mapping between domain statements and queries.

This repository contains IN and OUT Data splits along with templatized domain statements for IN split.

We follow the same Licensing Terms as BirdSQL dataset (i.e. Note: This dataset is constructed and distributed for academic use instead of commercial use.)
