use:
go build main.go
./main "127.0.0.1:2379" n startkey1 startkey2 .... startkeyn endkey1 endkey2 ... endkeyn

watch :
watch -c [-n] [] ./main "127.0.0.1:2379" n startkey1 startkey2 .... startkeyn endkey1 endkey2 ... endkeyn