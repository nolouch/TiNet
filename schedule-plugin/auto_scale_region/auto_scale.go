// Copyright 2018 PingCAP, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"sync"

	"github.com/gorilla/mux"
	"github.com/pingcap/log"
	"github.com/pingcap/pd/pkg/apiutil"
	"github.com/pingcap/pd/server/core"
	"github.com/pingcap/pd/server/schedule"
	"github.com/pingcap/pd/server/schedule/filter"
	"github.com/pingcap/pd/server/schedule/operator"
	"github.com/pingcap/pd/server/schedule/opt"
	"github.com/pingcap/pd/server/schedule/selector"
	"github.com/unrolled/render"
	"go.uber.org/zap"
)

// SchedulerType return type of the scheduler
func SchedulerType() string {
	return "auto-scale-region"

}

// SchedulerArgs provides parameters for the scheduler
func SchedulerArgs() []string {
	args := []string{"1"}
	return args

}

func init() {
	schedule.RegisterSliceDecoderBuilder("auto-scale-region", func(args []string) schedule.ConfigDecoder {
		return func(v interface{}) error {
			conf := v.(*scaleRegionConfig)
			conf.StartKeys = []string{"test"}
			conf.EndKeys = []string{"test"}
			return nil
		}
	})

	schedule.RegisterScheduler("reject-region", func(opController *schedule.OperatorController, storage *core.Storage, decoder schedule.ConfigDecoder) (schedule.Scheduler, error) {
		conf := &scaleRegionConfig{}
		decoder(conf)
		return newAutoScaleRegionScheduler(opController, storage, conf), nil
	})
}

const scaleRegionSchedulerName = "auto-scale-region-scheduler"

type scaleRegionConfig struct {
	sync.RWMutex
	storage   *core.Storage
	StartKeys []string `json:"start-key"`
	EndKeys   []string `json:"end-key"`
	rd        *render.Render
}

func (s *scaleRegionConfig) ListConfig(w http.ResponseWriter, r *http.Request) {
	s.RLock()
	defer s.RUnlock()
	s.rd.JSON(w, http.StatusOK, s)
}

func (s *scaleRegionConfig) UpdateConfig(w http.ResponseWriter, r *http.Request) {
	var input map[string]interface{}
	if err := apiutil.ReadJSONRespondError(s.rd, w, r.Body, &input); err != nil {
		return
	}
	var (
		startKeys []string
		endKeys   []string
	)
	s.Lock()
	defer s.Unlock()
	for i := 0; i < len(input)/2; i++ {
		startKeyTag := fmt.Sprintf("start-key-%d", i)
		endKeyTag := fmt.Sprintf("en-key-%d", i)
		startKey, ok := input[startKeyTag].(string)
		if !ok {
			s.rd.JSON(w, http.StatusInternalServerError, errors.New("arguments is valid"))
			return
		}
		startKeyBytes, err := hex.DecodeString(startKey)
		if err != nil {
			s.rd.JSON(w, http.StatusInternalServerError, err)
			return
		}
		endKey, ok := input[endKeyTag].(string)
		if !ok {
			s.rd.JSON(w, http.StatusInternalServerError, errors.New("arguments is valid"))
			return
		}
		endKeyBytes, err := hex.DecodeString(endKey)
		if err != nil {
			s.rd.JSON(w, http.StatusInternalServerError, err)
			return
		}
		startKeys = append(startKeys, string(startKeyBytes))
		endKeys = append(endKeys, string(endKeyBytes))
	}
	if len(startKeys) > 0 {
		s.StartKeys = startKeys
		s.EndKeys = endKeys
		s.rd.JSON(w, http.StatusOK, "susscess")
	}
}

type scaleRegionScheduler struct {
	*userBaseScheduler
	name     string
	config   *scaleRegionConfig
	selector *selector.BalanceSelector
	router   http.Handler
}

// LabelRegionScheduler is mainly based on the store's label information for scheduling.
// Now only used for reject leader schedule, that will move the leader out of
// the store with the specific label.
func newAutoScaleRegionScheduler(opController *schedule.OperatorController, storage *core.Storage, conf *scaleRegionConfig) schedule.Scheduler {
	filters := []filter.Filter{}
	kind := core.NewScheduleKind(core.RegionKind, core.BySize)
	base := newUserBaseScheduler(opController)
	conf.storage = storage
	conf.rd = render.New(render.Options{IndentJSON: true})
	router := mux.NewRouter()
	router.HandleFunc("/config", conf.UpdateConfig).Methods("POST")
	router.HandleFunc("/list", conf.ListConfig).Methods("GET")
	return &scaleRegionScheduler{
		name:              scaleRegionSchedulerName,
		config:            conf,
		userBaseScheduler: base,
		selector:          selector.NewBalanceSelector(kind, filters),
		router:            router,
	}
}

func (s *scaleRegionScheduler) EncodeConfig() ([]byte, error) {
	return schedule.EncodeConfig(s.config)
}

func (s *scaleRegionScheduler) GetName() string {
	return s.name
}

func (s *scaleRegionScheduler) GetType() string {
	return "reject-region"
}

func (s *scaleRegionScheduler) Cleanup(cluster opt.Cluster) {
}

func (s *scaleRegionScheduler) IsScheduleAllowed(cluster opt.Cluster) bool {
	return s.opController.OperatorCount(operator.OpRegion) < cluster.GetRegionScheduleLimit()
}

func (s *scaleRegionScheduler) Schedule(cluster opt.Cluster) []*operator.Operator {
	// schedulerCounter.WithLabelValues(s.GetName(), "schedule").Inc()
	stores := cluster.GetStores()
	targetStores := make(map[uint64]struct{})
	sourceStores := make(map[uint64]struct{})
	for _, s := range stores {
		if cluster.CheckLabelProperty(opt.RejectRegion, s.GetLabels()) {
			targetStores[s.GetID()] = struct{}{}
		} else {
			sourceStores[s.GetID()] = struct{}{}
		}
	}

	if len(sourceStores) == 0 || len(targetStores) == 0 {
		//	schedulerCounter.WithLabelValues(s.GetName(), "skip").Inc()
		return nil
	}

	log.Debug("label scheduler reject leader store list", zap.Reflect("stores", sourceStores))
	s.config.RLock()
	defer s.config.RUnlock()
	i := rand.Intn(len(s.config.StartKeys))
	var srcRegion *core.RegionInfo
	for _, region := range cluster.ScanRegions([]byte(s.config.StartKeys[i]), []byte(s.config.EndKeys[i]), 1024) {
		regionStores := region.GetStoreIds()
		for targetStore, _ := range targetStores {
			if _, ok := regionStores[targetStore]; !ok {
				srcRegion = region
				break
			}
		}
		if srcRegion != nil {
			break
		}
	}
	if srcRegion == nil {
		return nil
	}

	op, err := operator.CreateMoveRegionOperator("auto-scate-region", cluster, srcRegion, operator.OpRegion, targetStores)
	if err != nil {
		log.Error("create operator meet error", zap.Error(err))
		return nil
	}
	return []*operator.Operator{op}
}

func (s *scaleRegionScheduler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.router.ServeHTTP(w, r)
}
