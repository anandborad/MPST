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

# Extracting Movie Genres from a Plot Synopsis
We choose to watch a movie to relax and get disconnected from the usual schedule. And we expect it to be good according to our taste, like and dislike. 

When we plan to take 2-3 hours out of our busy schedule, we want that to be utilized completely for relaxation or entertainment. To achieve so, we go by the filters like actors, director, genres and more. Using those filters will help us to set an expectation as well as help us to choose a movie to watch.

Genres in movies are based on similarities either in narrative elements or in the emotional response to the movie. We can have a good idea about narrative elements and possible emotional response just by analyzing a movie plot synopsis.

In this blog, we will discuss a problem of movie synopsis analysis to extract genre(s) of a movie. For this exercise, we will be using a dataset provided by RiTUAL (Research in Text Understanding and Analysis of Language) Lab. More information is available here: http://ritual.uh.edu/mpst-2018/.

Such automated genre extracting system will also help to build better recommendation systems to predict similar movie and help us to know what to expect from a movie. This dataset contains around 14k movie synopsis derived into train, validation and test set. All the plots are categorized into one or multiple genres. Here, there is a total of 71 unique genres.

We would first import the dataset into a pandas data frame. The data can be downloaded from here: https://www.kaggle.com/cryptexcode/mpst-movie-plot-synopses-with-tags. Below is how dataset looks like in raw form:
