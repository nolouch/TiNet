package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/pingcap/tidb/kv"
	"github.com/pingcap/tidb/tablecodec"
	"github.com/pingcap/tidb/util/codec"
)

type Key struct {
	Desc        string `json:"desc"`
	Ts          uint64 `json:"ts,omitempty"`
	TableID     int64  `json:"table_id,omitempty"`
	RowID       int64  `json:"row_id,omitempty"`
	RowValue    int64  `json:"row_value,omitempty"`
	IndexID     int64  `json:"index_id,omitempty"`
	IndexValues string `json:"index_values,omitempty"`
}

func decodeKey(key string) Key {
	var ts uint64
	v, err := hex.DecodeString(key)

	if err != nil {
		panic(err)
	}
	tsString, decode, _ := codec.DecodeBytes(v, nil)
	if len(tsString) == 8 {
		_, ts, _ = codec.DecodeUintDesc(tsString)
	}
	if len(decode) > 0 && decode[0] == 'z' {
		decode = decode[1:]
	}
	desc := string(decode)
	tableID, indexID, isRecord, _ := tablecodec.DecodeKeyHead(kv.Key(desc))
	var (
		rowID       int64
		indexValues []string
	)
	if isRecord {
		_, rowID, _ = tablecodec.DecodeRecordKey(kv.Key(desc))
	} else {
		_, _, indexValues, _ = tablecodec.DecodeIndexKey(kv.Key(desc))
	}

	indexV := "\"\""
	if len(indexValues) != 0 {
		row_id, err := strconv.ParseInt(indexValues[0][2:], 10, 64)
		if err == nil {
			rowID = row_id
		}
		indexV = indexValues[1][2:]
	}

	return Key{
		Desc:        key,
		Ts:          ts,
		TableID:     tableID,
		RowID:       rowID,
		IndexID:     indexID,
		IndexValues: indexV,
	}
}

type Peer struct {
	Id        uint64 `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	StoreId   uint64 `protobuf:"varint,2,opt,name=store_id,json=storeId,proto3" json:"store_id,omitempty"`
	IsLearner bool   `protobuf:"varint,3,opt,name=is_learner,json=isLearner,proto3" json:"is_learner,omitempty"`
}

type RegionInfo struct {
	ID       uint64  `json:"id"`
	StartKey string  `json:"start_key"`
	EndKey   string  `json:"end_key"`
	Peers    []*Peer `json:"peers,omitempty"`
	Leader   *Peer   `json:"leader,omitempty"`
}

type RegionsInfo struct {
	Count   int           `json:"count"`
	Regions []*RegionInfo `json:"regions"`
}

type StoreInfo struct {
	Store struct {
		ID uint64 `json:"id"`
	} `json:"store"`
}

type StoresInfo struct {
	Stores []*StoreInfo `json:"stores"`
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("args error")
		return
	}
	pd := os.Args[1]
	res, err := http.Get(pd + "/pd/api/v1/stores")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer res.Body.Close()
	var stores StoresInfo
	err = json.NewDecoder(res.Body).Decode(&stores)
	if err != nil {
		fmt.Println(err)
		return
	}

	res, err = http.Get(pd + "/pd/api/v1/regions")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer res.Body.Close()
	var regions RegionsInfo
	err = json.NewDecoder(res.Body).Decode(&regions)
	if err != nil {
		fmt.Println(err)
		return
	}

	print_tail(stores.Stores, regions.Regions)

	var min, max string = " ", " "

	if len(os.Args) > 2 {
		l, err := strconv.Atoi(os.Args[2])
		if err != nil {
			fmt.Println("the three args err", err)
			return
		}

		for i := 0; i < l; i++ {
			min = os.Args[3+i]
			max = os.Args[3+i+l]
			print(stores.Stores, regions.Regions, min, max, 1)
		}
	} else {
		print(stores.Stores, regions.Regions, min, max, 0)
	}
}

func print_tail(stores []*StoreInfo, regions []*RegionInfo) {
	sort.Slice(regions, func(i, j int) bool { return regions[i].StartKey < regions[j].StartKey })
	var maxRegionIDLen int
	for _, r := range regions {
		if l := fieldLen(r.ID); l > maxRegionIDLen {
			maxRegionIDLen = l
		}
	}
	sort.Slice(stores, func(i, j int) bool { return stores[i].Store.ID < stores[j].Store.ID })
	var storeLen []int
	for _, s := range stores {
		storeLen = append(storeLen, fieldLen(s.Store.ID))
	}

	field(3, "▀", "\u001b[31m")
	field(10, "is_leader", "")
	field(3, "▀", "\u001b[33m")
	field(12, "is_learner", "")
	field(3, "▀", "\u001b[34m")
	field(8, "others", "")
	field(50, "key{tableID,rowID,indexID,indexValues}", "")
	fmt.Println()

	field(maxRegionIDLen, "", "")
	for i := range stores {
		field(storeLen[i], "S"+strconv.FormatUint(stores[i].Store.ID, 10), "")
	}
	field(25, "startkey ", "")
	field(25, "endkey", "")
	fmt.Println()
}

func print(stores []*StoreInfo, regions []*RegionInfo, min string, max string, kind int) {
	fmt.Println(min, "              ", max)
	var maxRegionIDLen int
	for _, r := range regions {
		if l := fieldLen(r.ID); l > maxRegionIDLen {
			maxRegionIDLen = l
		}
	}
	var storeLen []int
	for _, s := range stores {
		storeLen = append(storeLen, fieldLen(s.Store.ID))
	}

	for _, region := range regions {
		if kind == 1 && (region.StartKey < min || region.StartKey > max) {
			continue
		}
		field(maxRegionIDLen, "R"+strconv.FormatUint(region.ID, 10), "")
	STORE:
		for i, s := range stores {
			if region.Leader != nil && s.Store.ID == region.Leader.StoreId {
				field(storeLen[i], "▀", "\u001b[31m")
				continue
			}
			for _, p := range region.Peers {
				if p.StoreId == s.Store.ID {
					if p.IsLearner {
						field(storeLen[i], "▀", "\u001b[33m")
					} else {
						field(storeLen[i], "▀", "\u001b[34m")
					}
					continue STORE
				}
			}
			field(storeLen[i], "", "")
		}
		startkey := decodeKey(region.StartKey)
		fmt.Printf("     %4d, %7d, %4d, %10s  ", startkey.TableID, startkey.RowID, startkey.IndexID, startkey.IndexValues)
		endkey := decodeKey(region.EndKey)
		fmt.Printf(" %4d, %7d, %4d, %10s", endkey.TableID, endkey.RowID, endkey.IndexID, endkey.IndexValues)

		fmt.Println()
	}
}

func convertKey(k string) string {
	b, err := hex.DecodeString(k)
	if err != nil {
		return k
	}
	d, ok := decodeBytes(b)
	if !ok {
		return k
	}
	return strings.ToUpper(hex.EncodeToString(d))
}

func fieldLen(f interface{}) int {
	return len(fmt.Sprintf("%v", f)) + 2
}

func field(l int, s string, color string) {
	slen := utf8.RuneCountInString(s)
	if slen > l {
		fmt.Print(s[:l])
		return
	}
	if slen < l {
		fmt.Print(strings.Repeat(" ", l-slen))
	}
	if color != "" {
		fmt.Print(color)
	}
	fmt.Print(s)
	if color != "" {
		fmt.Print("\u001b[0m")
	}
}

// TiPD

func decodeBytes(b []byte) ([]byte, bool) {
	var buf bytes.Buffer
	for len(b) >= 9 {
		if p := 0xff - b[8]; p >= 0 && p <= 8 {
			buf.Write(b[:8-p])
			b = b[9:]
		} else {
			return nil, false
		}
	}
	return buf.Bytes(), len(b) == 0
}
