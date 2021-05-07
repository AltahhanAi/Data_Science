# The data quality and its structure

!!! success "Learning outcomes:"
	After completing this lesson you should be able to:

    * understand the difference between structured and unstructured data

    * know how to query both structured and unstructured data.

**Data is the blood stream of intelligence; it carries with it the power to capture the intricate details of the internal and external mechanism of a complex process. Yet it can be simpler and easier to deal with than directly trying to come up with models of even physical phenomena.**

However, the data can come in many different forms and stats. The integrity and accuracy of the data is problematic, and often a sanity check (checking the data makes sense) and cleaning (removing inconsistencies) need to be employed in order to be able to utilise the data. It may be that something called data wrangling is undertaken, where the structure of the data is changed to better suit the task.  

The data can come in different forms. For example, it might be static, meaning it does not change with time and it can be stored permanently without changing it. This is currently the prevailing dataset types that you will come across, at least in the context of learning about data mining.  

However, the data might come in a more dynamic form. For example, you may need to deal with data that is generated from a continuous stream, such as data that is dealing with sensors reading of a patient. Or a stream of tweets regarding a specific event that need to be processed in real-time to deal with ethical or security issues that might arise or to from an opinion of sentiment about the event that will be utilised by a market.  

##Structured, semi-structured and unstructured datasets

**In the previous lesson, you explored categorising data using the data mining model. Similarly, there are several other ways data can be categorised.**

One way is to look at how the data is stored, particularly whether the data is already stored on a database or not. Storing the data in a database implies that there is a specific structure in the database. This is often found in relational databases, but recently other types of database are emerging in the industry, such as non-SQL and Object-Oriented Databases, or hybrid databases.  

Having a specific structure in the database provides stronger guarantees of the sanity and quality of the data, which can be taken advantage of in the data mining processing task. For example, relational databases such as Oracle database is managed by Oracle relational database management system (RDBMS). Other examples include MySQL, which is a free open-source database acquired by Oracle and MS SQL Server which is owned by Microsoft. Such databases are often found in specific industries such as finance or retail, who were early adopters of database technologies and invested into making their data follow such structures. These industries have lots of legacy data that are stored in a database and need to be maintained in this format. These industries were the first adopters of database technologies because they needed a structure for their data, and the procedures in place today did not exist at the time.

However, most data is stored in a non-structural form, it comes in several looser forms such as text files or webpages. Extensible mark-up language or XML is a semi-structured data format that uses a set of rules and tags to define a structure in a document. In contrast to a database that needs a special engine to be read, it can be read by both humans and machines. It is an attempt to add some structure to the data utilised on the web and to separate the data from style formatting that used to be adopted in HTML. HTML5 is a move towards the semantic web era which attempts to deal with the World Wide Web as a giant architecture that is intended to provide answers and be more responsive and dynamic than its early HTML predecessor.

A dataset is a collection of data that has been formulated as a table or data frames representing a holistic or partial view of physical or virtual objects, processes or phenomena. The motivation behind collecting the data and storing it in one place (the dataset) is to prepare it for further analysis. Therefore, there is some engineering that will be involved in storing and preparing the data to make it suitable for the analysis task. However, this does not guarantee that the process can be automated fully. Please note, the term ‘database’ will be used to refer to data storage with an imposed, clear structure, while the term ‘data set’ will be used to refer to a collection of data in general that might not have a rigid or clear structure to it.

##XML, JSON, CSV and excel files

**Dealing with semi-structured and unstructured data comes with various challenges and opportunities.**

The opportunities are obvious; the more data that can be incorporated, the better the results of the task. The challenges are the diversity and unpredictability of the semi-structured and unstructured data. This module will deal with semi-structured and unstructured data most of the time, as this is most likely what you will come across in your work.

###JSON files

An example of semi-structured data is a JavaScript Object Notation (JSON) files, which is an alternative to XML. It is a minimal and readable format for slightly structuring the data. The main addition is that there is a pre-defined minimal set of types that we can utilise to add a bit of structure to our data. In a sense XML is a more general but looser way of defining data.

###CSV files

A less structured alternative to JSON is a comma-separated value (CSV) file. CSV files are comma delimited text files – a text file that has commas added between its entries to separate them. There are some tools that allows data fields to be added to a CSV file but in its original form there are no field types. These are intended to provide the least imposed structure but are intended to be dealt with like a text file; there is some structure giving the field names that are being dealt with without specifying the types of the fields. The type needs to be defined, at least whether they are numerical or non-numerical fields, when the data is processed.  

To summarise, a database:

* has a set of tables within a relationship
* can enforce integrity between the tables
* has strongly typed fields
* can have specified quality checking and data entry conditions and criteria.

XML files:

* are looser
* have attributes that can be defined that represent anything and not necessarily a field, including field type
* are open to convention
* allow anyone to specify any set of attributes that suit their needs.

JSON:

* has simpler structure than XML and provide less flexibility
* is basically a text file with predefined data types.

CSV files:

* are the least structured types of file
* are essentially a text file with commas to separate the different fields values.

A txt file:

* is simply a text file with no imposed structure, so perceivably a process might start by separating fields by a comma and then another process by a space or semi colon.

###Excel files

Inconsistencies and other issues arise when data is read into a specific algorithm for processing. Microsoft Excel files provide a built-in structure and ways to migrate to different formats. The result depends on the file content.  

It worth mentioning here that the data can be structured either in a loose way similar to a JSON or CSV file, or strongly structured when tables are used within Excel as there are plenty of inner links to give the data a stronger structure. However, a database is still more structured and provides more capabilities if the goal is to add more integrity and structure to the data.  

One positive aspect of Excel which contributed to its popularity is that it provides simple ways to see the data along with its visualisation (via charts, graphs, etc.). Having said that, you must differentiate between providing a storage capability for your data and acting on it in terms of processing and visualisation. If visualisation is important in the application, then Tableau provides such capabilities in an excellent way. You will see some examples of data visualisation at the end of this unit.

In the following exercises, you will see some of the most common data management activities in a structured data (database) and a similar activity in a non-structural data. These types of operations are normally done during the data engineering process. The first activity might be harder to set up and is optional, while the second activity is more important and easier to setup.

!!! abstract "Exercise"
		###Data querying, aggregation and summary: semi-structured and unstructured data.

		Try this <a href="https://minerva.leeds.ac.uk/bbcswebdav/xid-18737872_4" target="_blank">Data Wrangling exercise</a> in Jupyter notebook.


##Data quality and issues due to data mining

**There can be several issues that affect the quality of the data, which in turn will affect the quality of the results of the data mining process. Some of the common key issues are:**

###Noise

Regardless of the way it is collected, data will always have some level of inaccuracies, even when it is automatic sensor readings. This is called the noise of the data. The less noise in the data the better the data quality is, but it is impossible to collect data with zero noise.

For example, if reading physical phenomena such as the temperature, there is always going to be some level of inaccuracy and estimation from taking the measurement, coming from the used devices themselves or from both. This is part of the life of a data scientist and needs to be accepted, albeit while trying to reduce the noise as much as possible.

The noise can be treated either during the collection process, by increasing the level of the accuracies of the used devices, or by accounting for these inaccuracies when building a model for the data. It is assumed there is some level of noise that will affect and offset the analysis and model predictive capabilities.  

The question is how much noise is there? This question by itself is difficult to address but it would raise the accuracies of the analysis to know exactly how much noise the data has. But the question itself contradicts with the data scientist’s abilities; had they been able to measure accurately how much inaccuracy there is, there would have been accuracy in the collection process.

###Data leakage

Data leakage is quite a common and sometimes hard to spot phenomena. It happens mainly when data that was supposed to be used by one process spills into a second process accidentally, that it was not supposed to be used in. The leakage can be direct and obvious but sometimes can be subtle and hard to spot.  

###Precision and bias of an attribute

It is important to note here that this is considering the precision and bias of the data, not the model of the data. Remember the model is used to predict something, or in general to achieve a data mining task such as classification or clustering. You will explore the accuracy and precision of classification, regression and clustering in later units.

The **precision** of a feature (taken from a measurement, for example) is the closeness of measurements to one another.

The **bias** is a systematic variation of measurements from the actual quantity being measured. There is an implicit cause and effect problem here, but it can be assumed that you have the actual quantity that is being measured – at least in theory.  

###Example

The weight of an item is 1g and you want to specify the precision and bias of your measurement for it via a scale. The measurement of the item can be taken several times with the following results:

* 1.015
* 0.990
* 1.013
* 1.001
* 0.986

(The measurements in a dataset are not normally taken at one time but there is still a precision, bias and error for that matter that will be in the data, which may be more difficult to specify).

In this case, the mean is 1.001 and so the bias is 1.001-1=0.001. While the precision is measured by the readings, standard deviation from the mean gives an indication of the spread of the data from the mean.  

Remember that the standard deviation is the squared root of the variance. Also, since there is a limited set of samples – not the entire population – then it is divided by N-1.

Variance = [(1.015-1.001)2 + (0.990-1.001) 2 + (1.013-1.001) 2 + (1.001-1.001) 2+ (0.986-1.001) 2]/(5-1) = [0.000196 + 0.000121 + 0.000144 + 0 + 0.000225]/4 = 0.0001715 hence SD ≈ 0.13

##Summary

**In this lesson you have explored the basic ideas of data quality and structural, semi-structural and non-structural data.**

In the next lesson you will consider data preparation and how to perform it to set the scene for further analysis.
