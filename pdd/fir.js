a = function() {
            function e(t) {
                var n = this;
                Object(l.a)(this, e),
                this.successLoggerParams = function(e) {
                    return {
                        op: "event",
                        sub_op: "search",
                        p_search: s()(e.p_search),
                        req_params: Object(O.a)(n.requestModelParams),
                        query: n.searchKey,
                        sort: n.sortType,
                        is_sort: n.isSort ? 1 : 0,
                        page_index: n.searchPageIndex,
                        page_size: n.size
                    }
                }
                ,
                this.request = function() {
                    var e = Object(o.a)(i.a.mark(function e(t) {
                        var r, a, o, c, s, l;
                        return i.a.wrap(function(e) {
                            for (; ; )
                                switch (e.prev = e.next) {
                                case 0:
                                    return r = t.requestContext,
                                    a = void 0 === r ? {} : r,
                                    e.next = 4,
                                    k.a.getAntiContent();
                                case 4:
                                    (o = e.sent) && (n.antiContent = o);
                                case 6:
                                    return n.formatRequestParams(),
                                    e.prev = 7,
                                    e.next = 10,
                                    Object(d.a)({
                                        req: a.__req,
                                        path: "search",
                                        params: n.requestModelParams
                                    }).get();
                                case 10:
                                    if (e.t0 = e.sent,
                                    e.t0) {
                                        e.next = 13;
                                        break
                                    }
                                    e.t0 = {};
                                case 13:
                                    return c = e.t0,
                                    s = b(c, {
                                        sort: n.sortType,
                                        mallFilterSelected: n.mallFilterSelected,
                                        page: n.searchPageIndex
                                    }) || {},
                                    l = n.successLoggerParams(c),
                                    n.trackingSearchEvent(l),
                                    s.loadSearchResultTracking = l,
                                    s.firstFilter = n.requestModelParams.filter || "",
                                    n.flip = c.flip,
                                    n.updateFlip(c.flip),
                                    n.searchPageIndex++,
                                    e.abrupt("return", s);
                                case 25:
                                    return e.prev = 25,
                                    e.t1 = e.catch(7),
                                    e.abrupt("return", Object(y.a)(e.t1));
                                case 28:
                                case "end":
                                    return e.stop()
                                }
                        }, e, null, [[7, 25]])
                    }));
                    return function(t) {
                        return e.apply(this, arguments)
                    }
                }();
                var r = (t = t || {}).list || [];
                this.loadedNum = r.length,
                this.isBack = t.isBack,
                this.isSort = t.isSort || !1,
                this.sortType = t.sortType || "default",
                this.size = t.size || m.g,
                this.searchKey = t.searchKey,
                this.searchPageIndex = 1,
                this.mallFilterSelected = !!t.mallFilterSelected,
                this.loadedNum && (this.searchPageIndex = Math.ceil((this.loadedNum - this.loadedNum % m.g + m.g) / m.g),
                this.searchPageIndex = this.searchPageIndex > 1 ? this.searchPageIndex : 2),
                this.flip = t.flip || "",
                this.requestModelParams = this.formatBaseParams(t)
            }
            return Object(u.a)(e, [{
                key: "formatBaseParams",
                value: function(e) {
                    var t = Object(r.a)({}, e.urlApiParams);
                    return t.list_id = e.listID,
                    t.sort = e.sortType || "default",
                    t.filter = this.getFilterParams(e),
                    t.q = encodeURIComponent(e.searchKey || ""),
                    e.isBack,
                    t
                }
            }, {
                key: "getFilterParams",
                value: function(e) {
                    e = e || {};
                    var t = []
                      , n = Object(p.a)({
                        selectedProperty: e.selectedProperty,
                        selectedPropertyTag: e.selectedPropertyTag,
                        brandFilter: e.brandFilter,
                        selectedPropertyRec: e.selectedPropertyRec
                    });
                    return e.customPrice ? t.push(["price", e.customPrice, "custom"].join(",")) : e.price && t.push(["price", e.price].join(",")),
                    e.actFilterSelectedID && t.push("promotion,".concat(e.actFilterSelectedID)),
                    e.mallFilterSelected && t.push("favmall"),
                    n && t.push(n),
                    t.join(";")
                }
            }, {
                key: "formatRequestParams",
                value: function() {
                    var e = this.requestModelParams || {};
                    e.page = this.searchPageIndex,
                    e.size = 1 === this.searchPageIndex ? this.size : m.g,
                    this.flip && (e.flip = this.flip),
                    this.antiContent && (e.anti_content = this.antiContent)
                }
            }, {
                key: "trackingSearchEvent",
                value: function(e) {
                    Object(v.a)(e)
                }
            }, {
                key: "updateFlip",
                value: function(e) {
                    Object(_.a)({
                        flip: e
                    })
                }
            }]),
            e
        }()