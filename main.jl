using VideoIO
using Statistics
using Dates
using FileIO
using Base.Threads
using Images
using ColorTypes


VIDEO_DIR = raw"M:\SecurityFootage\FrontDoor\testdir"

SAMPLE_STRIDE = 30 # Use every Nth frame for median background
MOTION_PERCENTILE = 99.5 # Frames above this percentile are motion TODO: unused
MOTION_THRESHOLD = 1.25 # Absolute motion score threshold.
DILATION_FRAMES = 60 # Expand motion by +/- this many frames
BLACK_FRAMES = 10 # Separator between events

#FILE HELPERS

function save_background(background, outfile)

    img = RGB{N0f8}.(
        background[:,:,1] ./ 255,
        background[:,:,2] ./ 255,
        background[:,:,3] ./ 255
    )

    save(outfile, img)

end

function get_video_files(dir)

    files = filter(
        f -> occursin(r"^front_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4$", f),
        readdir(dir)
    )

    sort!(files)

    return joinpath.(dir, files)

end

function archivefiles(files,archive_dir)
    if isempty(files)
        return
    end
    mkpath(archive_dir)

    for file in files
        dest = joinpath(archive_dir, basename(file))
        mv(file, dest)
        println("Archived: $(basename(file))")
    end
end

function parse_timestamp(filename)
    m = match(
        r"front_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.mp4",
        basename(filename)
    )

    datestr = m.captures[1]
    timestr = replace(m.captures[2], "-" => ":")
    DateTime(Date(datestr), Time(timestr))
end

#FRAME UTILS

function frame_to_uint8(frame)
    h, w = size(frame)[1:2]
    out = Array{UInt8}(undef, h, w, 3)

    @inbounds for y in 1:h
        for x in 1:w

            p = frame[y,x]
            #println(p.r, p.g, p.b)
            out[y,x,1] = UInt8(round(Int, 255 * p.r))#TODO: check if this is the correct multiplier
            out[y,x,2] = UInt8(round(Int, 255 * p.g))
            out[y,x,3] = UInt8(round(Int, 255 * p.b))

        end
    end

    return out
end

#BACKGROUND CREATION
function fast_mode_background(v,counts)

    @inbounds for x in v
        counts[x+1] += 1
    end

    acc = 0
    target = length(v) ÷ 2

    for i in 1:256
        acc += counts[i]
        if acc > target
            return UInt8(i-1)
        end
    end
end

function build_background(video_file)
    println("Building median background: $video_file")
    reader = VideoIO.openvideo(video_file)

    samples = Vector{Array{UInt8,3}}()

    idx = 0

    while !eof(reader)

        frame = read(reader)

        idx += 1

        if idx % SAMPLE_STRIDE == 0
            push!(samples, frame_to_uint8(frame))
        end

    end

    close(reader)
    nsamples = length(samples)
    @assert nsamples > 0

    h,w,c = size(samples[1])
    background = Array{UInt8}(undef,h,w,c)
    println("Computing median image...")
    tmp = Vector{UInt8}(undef, nsamples)

    @inbounds for y in 1:h
        for x in 1:w
            for ch in 1:3

                for k in 1:nsamples
                    tmp[k] = samples[k][y,x,ch]
                end

                background[y,x,ch] = round(UInt8, median(tmp))

            end
        end
    end

    return background
end

function build_background_frames(frames)
    println("Building median background from frames")
    sampleidx = 1:SAMPLE_STRIDE:length(frames)
    nsamples = length(sampleidx)
    @assert nsamples > 0
    out = similar(frames[1])
    fourDtmp = Array{UInt8}(undef, nsamples, size(frames[1])...)
    for k in 1:nsamples
        fourDtmp[k,:,:,:] = frames[sampleidx[k]][:,:,:]
    end
    return round.(UInt8, median(fourDtmp, dims=1))[1,:,:,:]
end


#MOTION SCORE
function motion_score(frame_u8, background;percentile=98, eps=1e-6)

    dr = abs.(Int16.(frame_u8[:,:,1]) .- Int16.(background[:,:,1]))
    dg = abs.(Int16.(frame_u8[:,:,2]) .- Int16.(background[:,:,2]))
    db = abs.(Int16.(frame_u8[:,:,3]) .- Int16.(background[:,:,3]))

    mean_r = mean(dr)
    mean_g = mean(dg)
    mean_b = mean(db)

    p_r = quantile(vec(dr), percentile/100)
    p_g = quantile(vec(dg), percentile/100)
    p_b = quantile(vec(db), percentile/100)

    score_r = p_r / (mean_r + eps)
    score_g = p_g / (mean_g + eps)
    score_b = p_b / (mean_b + eps)

    return (score_r + score_g + score_b) / 3
end
function motion_score_medium(frame_u8, background;percentile=50, eps=1e-6)

    diff = vec(sum(abs, frame_u8 .- background, dims = 3) ./ 3)

    p = quantile(diff, percentile/100)
    #m = mean(diff)
    return p #/ (m + eps)
end
 
function motion_score_simple(frame_u8, background)
    #println(typeof(frame_u8),typeof(background))
    return sum(abs.(Int16.(frame_u8) .- Int16.(background)))
end

function motion_score_fast(frame_u8, background; pix=1000)

    #println(size(frame_u8))
    #println(size(background))
    diff = vec(sum(abs, frame_u8 .- background, dims = 3))

    k = pix
    pivot = length(diff) - k + 1

    partialsort!(diff, pivot; rev=true)

    return mean(@view diff[pivot:end])/3
end

function dilate_motion(motion::BitVector)
    out = copy(motion)

    for idx in findall(motion)
        lo = max(1, idx - DILATION_FRAMES)
        hi = min(length(out), idx + DILATION_FRAMES)

        @views out[lo:hi] .= true
    end

    return out
end

#EVENT EXTRACTION
function find_event_ranges(motion)
    ranges = UnitRange{Int}[]
    active = false
    startidx = 0

    for i in eachindex(motion)
        if motion[i] && !active
            startidx = i
            active = true

        elseif !motion[i] && active
            push!(ranges, startidx:(i-1))
            active = false

        end
    end

    if active
        push!(ranges, startidx:length(motion))
    end

    return ranges
end

#MAIN
files = get_video_files(VIDEO_DIR)

isempty(files) && error("No matching files found.")

mkpath(joinpath(VIDEO_DIR, "motion_events"))

start_time = parse_timestamp(first(files))
end_time = parse_timestamp(last(files)) + Minute(5)

output_name = Dates.format(start_time, "yyyy-mm-dd_HH-MM-SS") * "_to_" * Dates.format(end_time, "yyyy-mm-dd_HH-MM-SS") * ".mp4"

output_path = joinpath(VIDEO_DIR, "motion_events", output_name)

writer = nothing
h,w,_ = size(frame_to_uint8(VideoIO.openvideo(first(files)) |> read))
fps = 25

writer =
    VideoIO.open_video_out(
    output_path,
    UInt8,
        (h,w),
        framerate=fps
    )

#Threads.@threads 
for file in files

    println()
    println("======================================")
    println(file)
    println("======================================")
    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Opening video...")
    reader = VideoIO.openvideo(file)
    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Reading frames...")#TODO: kinda slow
    frames = Vector{Array{UInt8,3}}()

    while !eof(reader)
        frame = read(reader)
        push!(frames, frame_to_uint8(frame))
    end
    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    background = build_background_frames(frames)
    #=
    bgdir = joinpath(VIDEO_DIR, "backgrounds")
    mkpath(bgdir)

    bgfile = joinpath(
        bgdir,
        replace(basename(file), ".mp4" => "_background.png")
    )

    save_background(background, bgfile)

    println("Saved background image:")
    println(bgfile) 
    =#

    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Computing motion scores...")
    scores = zeros(Float64, length(frames))
    sumvalues = 0.0
    for i in 1:1:length(frames)
        frame = frames[i]
        scores[i] = motion_score_simple(frame, background)
        #sumvalues += mean(frame)
    end
    #println(scores)
    #avgbrightness = sumvalues/length(frames)

    close(reader)
    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Computing motion threshold...")
    println("Maximum score: $(maximum(scores))")
    println("Average score: $(mean(scores))")
    thresh = MOTION_THRESHOLD*mean(scores)#TODO: calculate this based on avgbrightness

    println("Motion threshold: ", thresh)

    motion = BitVector(score > thresh for score in scores)
    motion = dilate_motion(motion)
    events = find_event_ranges(motion)

    println(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println("Events found: ", length(events))
    println("writing events to output video...")
    black = zeros(UInt8,size(frames[1]))
    for ev in events
        println("Frames $(first(ev)) -> $(last(ev))")

        for idx in ev
            write(writer, frames[idx])
        end

        for _ in 1:BLACK_FRAMES
            write(writer, black)
        end
    end

end

if writer !== nothing
#close(writer)
VideoIO.close_video_out!(writer)
end

println()
println("Archiving source files...")
archivefiles(files, joinpath(VIDEO_DIR, "archived_videos"))
println()
println("Done.")
println(output_path)