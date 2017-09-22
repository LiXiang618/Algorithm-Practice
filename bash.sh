#!/bin/bash
git add .
git commit -m $(echo -e Updated\\$(date | grep -o -E '[0-9]+') | tr ' ' '-')
git push origin master