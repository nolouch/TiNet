from django.http import HttpResponse
import json
from . import globalvar

def regions(request):
    return HttpResponse(json.dumps(globalvar.get_regions(), ensure_ascii=False), content_type="application/json,charset=utf-8")
def tikv_replicas(request):
    return HttpResponse(json.dumps(globalvar.get_tikv_replicas(), ensure_ascii=False), content_type="application/json,charset=utf-8")
def tidb_replicas(request):
    return HttpResponse(json.dumps(globalvar.get_tidb_replicas(), ensure_ascii=False), content_type="application/json,charset=utf-8")