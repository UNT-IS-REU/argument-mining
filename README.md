# Argument Mining

---
## Project Tasks
Here is a brief introduction of the background and tasks of the project:
 
> The data we will use is the reported (or published) court report. Each report is a summarization of how a court judges one case, including several reasoning components such as the facts, the rules or laws used, the reasoning process, and the final decision. An argument is a sentence (maybe part of a sentence or more than one sentence, in our current project, we take it as a sentence) belonging to a category in the reasoning process, and naturally, there can be directional relationships of arguments in the reasoning process, e.g., a fact may be for or against an issue. In this regard, each reported case is a comprehensive structure of arguments, or more specifically, a graph or tree of arguments in which each node is an argument, and each edge is a directional relationship.
 
> The ultimate goal of the project is to automatically generate the complete reasoning structure according to a given set of facts. We divide the task into two parts: building an argument database according to existing cases, and building a retrieval system to automatically find & generate the report-like argument skeleton based on the given facts. Due to the large volume of data and the need for automation, we employ classification techniques in building the database and retrieving related arguments. Well-labeled ground truth is essential for the classification, and therefore for the whole task.
 
> In the above settings, our current task is to generate high-quality annotation of arguments and their structures for several legal reports. We assume there are five categories of arguments: facts, rule or law, issue, analysis, and conclusion, and the arguments appear in a tree structure. For each court report, we will need to manually judge which argument category each of its sentence belongs to, and whether there is any relationship (for, against, or no relationship) between the labeled argument sentences.
 
> The evaluation should be intrinsic since we have no ground truth. We will evaluate the annotator’s agreement level to judge the annotation quality. It is necessary to note that we also need to explore what agreement measure works best for our case and how to improve the quality. Since the annotation involves high-level semantics, the label can be pretty vague sometimes, and the agreement level can be low as a result. Quality control should also be part of our research task.
 
> As for the dataset, we have several cases for WestLaw and Courtlistener. It is also a great option to download some data recently released by the Harvard law library.

[![Run on Repl.it](https://repl.it/badge/github/UNT-IS-REU/argument-mining)](https://repl.it/github/UNT-IS-REU/argument-mining)
