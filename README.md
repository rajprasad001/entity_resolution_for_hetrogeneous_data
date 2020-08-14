# Entity Resolution for Hetrogeneous Data
Entity resolution is one of the key steps involved in data integration, identifying and linking different manifestations of records in different data sources to the same real-world entity.
The dataset for the task comprised of 29897 entities in the form Json files comprising the product description of cameras. The data was extracted from different e-commerce websites. This made the data hetrogeneous in terms of schema. 
Along with the dataset, a groundtruth was provided comprising match and non-match pairs of 928 entities which were subset of the main dataset.

Our task was to predict the match pairs in the main dataset using Machine learning concept with the help of Ground truth provided.

The general pipeline of the task is mentioned below:-

* Understanding the data and finding hidden patterns for selecting the best suitable features for further use.

  Each Json file comprised details of the product. Though the schema varied throughout the dataset, there were few features like 'Pagetitle'.
  A pagetitle had summarized information of the product with some unwanted text like marketing offers. Since Pagetitle was common in all files, we selected this as a primary
  feature for our model. Apart from the pagetitle, we used regex model to extract models and brands for all camera entities which were addtional features for our model.

* Filtering Non-products
  In the dataset, we encountered almost 3000 non-camera ids which we filtered using weak supervision techniques. This was implemented with the help of a powerful tool called Snorkel.
  
  
* Defining the pipeline for vector representation of features

  The selected features(Pagetitle, Model and Brand) needed to be converted into a vector space so as it can be used for machine learning models. For this following techniques 
  were used:-
  
  * Tokenising
  * Lowercase
  * Stopwords removal
  * Embedding the pagetitle with the help of fasttext library
  * Binary encoding of both Models and Brands
  
 * Blocking(Hashing)
 
    Blocking is an effective method to reduce the amount of computation required due to comparision of each element with every element in the dataset to make pairs.
    The idea behind blocking is that elements with same hashcodes are kept in the same bucket and the comparision to form pairs is only done between the elements of a bucket.
    This drastically reduces the number of possible pairs. The blocking results are evaluated on two parameters:-
  
    * Computation
    * Coverage
  
 * Forming Pairs and Training the Model
 
    The buckets formed after blocking was used to generate possible pairs (P).
    Further, on groundtruth different models were trained and testing results were observed. The best model not suceptible to overftting was used to assign the possible pairs 
    (P)to either a match or a non-match.
    The input to the model was an absolute vector difference of the pairs.
  
  * Consistency Check
  
    Inorder to expand the results, and number of match or non-match, consistency check was perfromed on the final result.
  
  *Final reults and detailed pipleine of the project are mentioned in the report*
