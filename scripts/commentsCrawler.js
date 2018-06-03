  // Output shape:
  // id, parentId, authorName, authorId, replies, upvotes, text, timestamp
  getCommentsMetaData = (c) => ([
    c.dataset.commentId,
    getCommentsParent(c),
    c.dataset.commentAuthor,
    c.dataset.commentAuthorId,
    c.dataset.commentReplies,
    c.querySelector('.d-comment__recommend').dataset.recommendCount,
    c.querySelector('.d-comment__body p').innerHTML,
    c.dataset.commentTimestamp,
  ])

  getCommentsParent = c => {
      const referrerElement = c.querySelector('.js-discussion-author-link');
      if (!referrerElement)
        return null;
      const referrer = referrerElement.href;
      const anchorPos = referrer.indexOf('#comment-')
      const parentId = referrer.slice(anchorPos + 9)
      if (parentId == c.dataset.commentId)
        return null;
      return parentId;
  }

  arrayToCSV = arr => {
    const fields = ['id', 'parentId', 'authorName', 'authorId', 'replies',
      'upvotes', 'text', 'timestamp']
    const replacer = (key, value) => (value === null ? '' : value)
    const csv = arr.map(
      row => row.map(
        field => JSON.stringify(field, replacer)
      ).join(',')
    )
    csv.unshift(fields.join(',')) // add header column
    return csv.join('\n').replace(/\\"/g, '""')
  }

  comments = Array.from(document.querySelectorAll('.d-comment'))
  comments = comments.map(getCommentsMetaData)
  csv = arrayToCSV(comments)
