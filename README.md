```
pip install -e .
```

```
$ echo '私は日本人です。' | mlm-score
100%|| 1/1 [00:01<00:00,  1.12s/it]
-0.5098785758018494     私は日本人です。
```

```
$ echo '私は人本日です。' | mlm-score
100%|| 1/1 [00:00<00:00,  1.42it/s]
-4.811020851135254      私は人本日です。
```

テキストのフィルタリングとかに使ってください。

