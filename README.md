## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/anandborad/MPST/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/anandborad/MPST/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

------------------------

# Extracting Movie Tags from a Plot Synopsis
We choose to watch a movie to relax and get disconnected from the usual schedule. And we expect it to be good according to our taste, like and dislike. 

When we plan to take 2-3 hours out of our busy schedule, we want that to be utilized completely for relaxation or entertainment. To achieve so, we go by the filters like actors, director, genres and more. Using those filters will help us to set an expectation as well as help us to choose a movie to watch.

Genres in movies are based on similarities either in narrative elements or in the emotional response to the movie. We can have a good idea about narrative elements and possible emotional response just by analyzing a movie plot synopsis.

In this blog, we will discuss a problem of movie synopsis analysis to extract genre(s) of a movie. For this exercise, we will be using a dataset provided by RiTUAL (Research in Text Understanding and Analysis of Language) Lab. More information is available here: [http://ritual.uh.edu/mpst-2018/](http://ritual.uh.edu/mpst-2018/).

Such automated genre extracting system will also help to build better recommendation systems to predict similar movie and help us to know what to expect from a movie. This dataset contains around 14k movie synopsis derived into train, validation and test set. All the plots are categorized into one or multiple genres. Here, there is a total of 71 unique genres.

We would first import the dataset into a pandas data frame. The data can be downloaded from [here](https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags). Below is how dataset looks like in raw form:

![MPST Data](https://raw.githubusercontent.com/anandborad/MPST/master/images/MPST%20data.png)

## 1. Understanding the dataset

From the above dataset, let's take a close look at different columns:

1. imdb_id: Internet Movie Database (IMDb) is the most popular and authoritative source to learn about movie or TV series. For every digital content, IMDb generates a unique identifier which is accepted all around the internet. Here, imdb_id is a unique identifier and should be unique for each datapoint. If there are duplicate imdb_id, that simply means we do have duplicates data-points in our data and we need to remove those.

2. title: the title of the movie
3. Plot_synopsis: Plot synopsis is the narrative explanation of a movie plot means a summary of a screenplay. It introduced a lead character and what it does in a movie. Below is the plot synopsis for the movie ‘Hansel and Gretel’ (imdb_id: tt1380833):

```
‘Hansel and Gretel are the young children of a poor woodcutter. When a great famine settles over the land, the woodcutter\'s second, abusive wife decides to take the children into the woods and leave them there to fend for themselves, so that she and her husband do not starve to death, because the kids eat too much. The woodcutter opposes the plan but finally, and reluctantly, submits to his wife\'s scheme. They were unaware that in the children\'s bedroom, Hansel and Gretel......’
```

4. split: this column defines whether a data-points belongs to a train, test or validation set.

5. synopsis_source: It gives information about the synopsis source, either IMDb or Wikipedia.

6. tags: Tags are labelled genres for a movie. It may take multiple values for a single movie. This will be our prediction labels.
If we take a closure look, a single tag may have space or a ‘-’. We want our tags to be similar form and thus we will replace whitespace and a dash with an underscore (‘_’). Also, we will separate tags by space instead of the comma. Below is how it looks:
