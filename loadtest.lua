-- DocRes load test: multipart file upload with random image selection
local images = {}
local boundary = "----DocResLoadTest"
local api_key = "f0601d1a53253f9f86729cb9ceacd452fb7d3f9b876c17146061a26fb95c20d5"

function read_file(path)
    local f = io.open(path, "rb")
    if not f then return nil end
    local data = f:read("*all")
    f:close()
    return data
end

function init(args)
    local dir = args[1] or "."
    local files = {
        "for_debluring.png",
        "for_deshadowing.jpg",
        "for_binarization.png",
    }
    for _, name in ipairs(files) do
        local data = read_file(dir .. "/" .. name)
        if data then
            table.insert(images, { name = name, data = data })
            io.write(string.format("Loaded: %s (%d KB)\n", name, #data / 1024))
        end
    end
    if #images == 0 then
        io.write("ERROR: no images loaded\n")
        os.exit(1)
    end
end

function request()
    local img = images[math.random(#images)]
    local body = "--" .. boundary .. "\r\n"
        .. 'Content-Disposition: form-data; name="file"; filename="' .. img.name .. '"\r\n'
        .. "Content-Type: application/octet-stream\r\n\r\n"
        .. img.data .. "\r\n"
        .. "--" .. boundary .. "--\r\n"

    return wrk.format("POST", nil, {
        ["Content-Type"] = "multipart/form-data; boundary=" .. boundary,
        ["Authorization"] = "Bearer " .. api_key,
    }, body)
end

local status_200 = 0
local status_err = 0
local errors = {}

function response(status, headers, body)
    if status == 200 then
        status_200 = status_200 + 1
    else
        status_err = status_err + 1
        errors[status] = (errors[status] or 0) + 1
    end
end

function done(summary, latency, requests)
    io.write("\n--- Results ---\n")
    io.write(string.format("OK (200):    %d\n", status_200))
    io.write(string.format("Errors:      %d\n", status_err))
    for code, count in pairs(errors) do
        io.write(string.format("  HTTP %d:    %d\n", code, count))
    end
    io.write(string.format("Avg latency: %.2f ms\n", latency.mean / 1000))
    io.write(string.format("P99 latency: %.2f ms\n", latency:percentile(99) / 1000))
    io.write(string.format("Requests/s:  %.2f\n", summary.requests / (summary.duration / 1000000)))
end
