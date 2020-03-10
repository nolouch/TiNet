from django.http import HttpResponse
import json
from . import globalvar

def data(request):
    return HttpResponse(json.dumps(globalvar.get_demo_value(), ensure_ascii=False), content_type="application/json,charset=utf-8")
