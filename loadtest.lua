-- DocRes load test: /full vs /full-dewarp split 50/50, resolution=2048, upscale=false
local boundary = "----DocResLoadTest"
local api_key = "f0601d1a53253f9f86729cb9ceacd452fb7d3f9b876c17146061a26fb95c20d5"
local image = nil

local endpoints = {
    "/full?resolution=2048&upscale=false",
    "/full-dewarp?resolution=2048&upscale=false",
}

-- counters per endpoint (index 1 = /full, 2 = /full-dewarp)
local counts = { { ok = 0, err = 0 }, { ok = 0, err = 0 } }
local errors = {}
local order = {}  -- FIFO of endpoint index per in-flight request

function read_file(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("*all")
    f:close()
    return data
end

function init(args)
    local dir = args[1] or "."
    local name = "for_deshadowing.jpg"
    image = { name = name, data = read_file(dir .. "/" .. name) }
    if not image.data then
        io.write("ERROR: cannot load " .. dir .. "/" .. name .. "\n")
        os.exit(1)
    end
    io.write(string.format("Loaded: %s (%d KB)\n", name, #image.data / 1024))
end

function request()
    local idx = math.random(2)
    table.insert(order, idx)
    local body = "--" .. boundary .. "\r\n"
        .. 'Content-Disposition: form-data; name="file"; filename="' .. image.name .. '"\r\n'
        .. "Content-Type: application/octet-stream\r\n\r\n"
        .. image.data .. "\r\n"
        .. "--" .. boundary .. "--\r\n"

    return wrk.format("POST", endpoints[idx], {
        ["Content-Type"] = "multipart/form-data; boundary=" .. boundary,
        ["Authorization"] = "Bearer " .. api_key,
    }, body)
end

function response(status, headers, body)
    local idx = table.remove(order, 1) or 1
    if status == 200 then
        counts[idx].ok = counts[idx].ok + 1
    else
        counts[idx].err = counts[idx].err + 1
        errors[status] = (errors[status] or 0) + 1
    end
end

function done(summary, latency, requests)
    io.write("\n--- Results ---\n")
    io.write(string.format("Total requests: %d\n", summary.requests))
    io.write(string.format("  /full         OK=%d  ERR=%d\n", counts[1].ok, counts[1].err))
    io.write(string.format("  /full-dewarp  OK=%d  ERR=%d\n", counts[2].ok, counts[2].err))
    local total_err = 0
    for code, count in pairs(errors) do total_err = total_err + count end
    io.write(string.format("Errors total: %d\n", total_err))
    for code, count in pairs(errors) do
        io.write(string.format("  HTTP %d:    %d\n", code, count))
    end
    io.write(string.format("Avg latency: %.2f ms\n", latency.mean / 1000))
    io.write(string.format("P99 latency: %.2f ms\n", latency:percentile(99) / 1000))
    io.write(string.format("Requests/s:  %.2f\n", summary.requests / (summary.duration / 1000000)))
end
