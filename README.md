# article_summarizer
Summarize an article with most relevant sentences and keywords


## Example
```python
import article_summarizer
article_summary = ArticleSummary(url="http://en.wikipedia.org/wiki/Content_analysis")
print(article_summary.article.title)
for i in xrange(0, 5):
    print(article_summary.summary[i]['raw_text'])
sorted_keywords = sorted(article_summary.keywords.items(), key=itemgetter(1), reverse=True)
for i in xrange(0, 5):
    print(sorted_keywords[i])
```
