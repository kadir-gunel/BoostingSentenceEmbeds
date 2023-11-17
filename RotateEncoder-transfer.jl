cd(@__DIR__)
using Pkg
Pkg.activate("/raid/Glowe")

using Logging
using Random
Random.seed!(1234)

using .Iterators
using Dates
using Printf
using LinearAlgebra
using Statistics: mean
using Test

using XLEs

using Flux
using CUDA
using Transformers
using Transformers.Layers
using Transformers.TextEncoders


using Flux.Losses
using Flux: gradient
using Flux.Optimise: update!
using Flux.Data: DataLoader

using BSON: @save, @load


CUDA.device!(1)
enable_gpu(CUDA.functional()) # make `todevice` work on gpu if available


experiment = :encoderRotated
data = "wmt-crawl-ft"
ups = :wups


timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
save_path = "./models/sentences/$(experiment)/$(data)/$(ups)/$(timestamp)";
!isdir(save_path) ? mkpath(save_path) : nothing
logfile = "$(save_path)/log.$(ups).$(timestamp).txt"

README = "THIS IS THE EXPERIMENT WHERE A TRANSFORMER ENCODER MODEL IS BEING USED.
          THE MODEL CONSISTS OF :
          1. WORD EMBEDDING LAYER
          2. A ENCODER: Encoder outputs all words of a sentence. So for a
          given batch which consists of BSIZE sentences. ENCODER outputs
          a D x L x B tensor. D being the dimension, L being the sentence
          length and B is the batch size. 
          3. MAKING DIMENSION EQUAL: A FFNN
          2nd setep is followed by a simple FFNN :
                 i. Takes the average of D x L x B tensor and outputs D x B
                 ii.The FFNN then converts this and outputs a 768 dimensional
                    vector for distance comparison.
          THIS EXPERIMENT CONTROLS 3 THINGS :
          1. EUCLIDEAN
          2. COSINE
          3. EUCLIDEAN + COSINE 
          IN THIS SET OF EXPERIMENTS, WE EXAMINE HOW WORD EMBEDDINGS CAN
          HAVE INFORMATION FROM SBERT SENTENCE EMBEDDINGS BY USING DISTANCE
          MINIMIZATION TECHNIQUE. 

          (IN ADDITION, WE ALSO KEEP TRACK OF PP EVENTHOUGH IT IS NOT VALID!)
          
         "



@info "Writing to logfile : $(logfile)"
io = open(logfile, "w+")


logger = SimpleLogger(io)
global_logger(logger)

@warn CUDA.device()
@info README


function loadEmbeddings(fname::String, corpusVoc::Vector{String})
    @info "Loading FastText Embedding: $fname"
    V, WE = readBinaryEmbeddings(fname)
    WE = WE |> permutedims
    s2i = Dict(word => i for (i, word) in enumerate(V))

    word_idx = collect(haskey(s2i, word) ? s2i[word] : -1 for word in corpusVoc)
    # -1 for unknown words
    isminusone(x) = x == -1
    filter!(!isminusone, word_idx)
    sort!(word_idx)

    return V[word_idx], WE[:, word_idx]
end


function createTextEncoder(V::Vector{String}, WE::Matrix)
    @info "Processing Word Embeddings and Creating Text Encoder"
    startsym = "<s>"
    endsym = "</s>"
    unksym = "<unk>"
    tags = [unksym, startsym, endsym]

    r = collect(findfirst(symbol .== V) for symbol in tags)
    k = r .== nothing # these should be added to vocabulary
    labels = sum(k) >= 1 ? vcat(tags[k], V) : V 
    
    textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

    @info "Adding Embeddings for the special chars in WE Space"

    for tag in tags[k]
        WE = tag == unksym ? hcat(mean(WE[:,  end-99:end], dims=2), WE) : WE        
        WE = tag == startsym ? hcat(WE[:, findfirst(("the") .== labels)], WE) : WE 
        WE = tag == endsym ? WE = hcat(mean(WE[:, 1:10], dims=2), WE) : WE 
    end
    flush(io)
    return textenc, WE
end


function embedding(input)
    we = M.WEMBEDS(input.token)
    pe = pos_embed(we)
    return we .+ pe
end

function encoder_forward(input)
    attention_mask = get(input, :attention_mask, nothing)
    e = embedding(input)
    t = M.ENCODER(e, attention_mask) # return a NamedTuples (hidden_state = ..., ...)
    h = reshape(mean(t.hidden_state, dims=2), hidden_dim, bsize) # |> augmentor
    return h
end

p_norm(M::T; dim=2) where {T} = sqrt.(sum(real(M .* conj(M)), dims=dim))
cosine_sim(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))')

cosine(X::T, Y::T) where {T} = 1 .- abs.(cosine_sim(X |> permutedims, Y |> permutedims))
euclidean(X::T, Y::T) where {T} = sqrt.(sum((X .- Y) .^ 2, dims=1))

function dloss(input, output; distance=euclidean)
    enc = encoder_forward(input)
    return mean(distance(enc, output))
end

function combinedloss(input, output; α::Real=0.5, β::Real=0.5)
    enc = encoder_forward(input)
    eloss = mean(euclidean(enc, output))
    closs = mean(cosine(enc, output))
    return (α * eloss) + (β * closs)
end


anynan(x) = any(y -> any(isnan, y), x)

function train!(; distance=:euclidean, epochs::Int=20)

    distance = (distance == :euclidean) || (distance == :cosine) ?
        eval(distance) : :both

    loss_min = typemax(Float32)
    last_improvement = 0
    trn_size = floor(length(sentences) * 0.95) |> Int

    train_data = DataLoader((sentences[1:trn_size], sBertRed[:, 1:trn_size]), batchsize=bsize,
                            partial=false, shuffle=false)
    test_data = DataLoader((sentences[1+trn_size:end], sBertRed[:, 1+trn_size:end]), batchsize=bsize,
                           partial=false, shuffle=false)

    """
    train_data = DataLoader((sentences[1:1000], sBertRed[:, 1:1000]), batchsize=bsize,
                            partial=false, shuffle=false)
    test_data = DataLoader((sentences[1+1000:2000], sBertRed[:, 1+1000:2000]), batchsize=bsize,
                           partial=false, shuffle=false)
    """

    for epoch in 1:epochs
        trn_losses = Float32[]; tst_losses = Float32[];
        dgrad = nothing; # either of euclidean or cosine
        egrad = nothing; # for euclidean
        cgrad = nothing; # for cosine
        for(gsents, sbert) in train_data
            input = encode(textenc, gsents) |> todevice
            if isequal(distance, :both) # either euclidean or cosine
                dgrad = gradient(()-> combinedloss(input, sbert |> todevice), ps)
            else
                egrad = gradient(()-> dloss(input, sbert |> todevice, distance=euclidean), ps)
                cgrad = gradient(()-> dloss(input, sbert |> todevice, distance=cosine), ps)
            end

            push!(trn_losses, combinedloss(input, sbert |> todevice))
            push!(trn_losses, dloss(input, sbert |> todevice, distance=euclidean))
            push!(trn_losses, dloss(input, sbert |> todevice, distance=cosine))

            if isequal(distance, :both)
                update!(opt, ps, dgrad) # this is batch update!
            else
                update!(opt, ps, egrad) # this is batch update!
                update!(opt, ps, cgrad) # this is batch update!
            end
            
        end

        com = mean(trn_losses[1:3:end])
        euc = mean(trn_losses[2:3:end])
        cos = mean(trn_losses[3:3:end])

        
        @info "Epoch: $(epoch) Combined Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos)"
        printstyled("Epoch: $(epoch)\n", color=:red)
        printstyled("Combined Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos) \n", color=:light_cyan)


        for(gsents, sbert) in test_data
            input = encode(textenc, gsents) |> todevice
            push!(tst_losses, combinedloss(input, sbert |> todevice))
            push!(tst_losses, dloss(input, sbert |> todevice, distance=euclidean))
            push!(tst_losses, dloss(input, sbert |> todevice, distance=cosine))
        end

        com = mean(tst_losses[1:3:end])
        euc = mean(tst_losses[2:3:end])
        cos = mean(tst_losses[3:3:end])

        @info "Test Losses: Combined Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos)"
        printstyled("Combined Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos) \n", color=:light_cyan)
        #if abs(mean(trn_losses) - loss_min) > 0.04
        #    model = Chain(word_embeds |> cpu, encoder |> cpu, augmentor |> cpu) |> cpu
        #    model = Flux.state(model)
        M_CPU = ENC(M.WEMBEDS |> cpu, M.ENCODER |> cpu)
        @save "$(save_path)/$(string(distance))/$(epoch).bson" M_CPU
        #    if distance == euclidean
        #        loss_min = mean(trn_losses[1:3:end]) # euclidean
        #    end
        #    if distance == cosine
        #        loss_min = mean(trn_losses[2:3:end]) # cosine
        #    end
        #    last_improvement = epoch
        # end

        if anynan(ps)
           @error "NaN params"
           break
        end

        #if epoch - last_improvement >= 5 && opt.eta > 1e-6
        #   opt.eta /= 10.0
        #   @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
        #   # After dropping learning rate, give it a few epochs to improve
        #   last_improvement = epoch
        #end

        #if epoch - last_improvement >= 10
        #   @warn(" -> We're calling this converged.")
        #   break
        #end

        flush(io) # writes to log file 
        
    end
end


flush(io) # writes all information till here  to log file 


@info "creating S2S model Object"
struct ENC
    WEMBEDS::Embed
    ENCODER::Transformer
end


dfile = "/home/phd/Documents/Journal/journal_wmt_selected_lines.bson"
@info "Reading Sentences : $(dfile)"
@load dfile shuffled_lines

sentences = shuffled_lines .|> String
corpusVoc = sentences .|> split |> flatten |> collect |> unique .|> String

bname =  "/home/phd/Documents/Journal/sBertRedVecs.bson"
@info "Loading sBert Reduced Sentence Vectors from : $(bname)"
@load bname sBertRed

flush(io)

# global λ = 1e-2
# global λ = 3e-4
# trn_list = [:euclidean, :cosine, :both]
trn_list = [:both]
for dist in trn_list
    mkdir(save_path * "/$(dist)")
    # fname = "./models/FT/$(data)"
    # fname = "/home/phd/Documents/Embeddings/$(data)"
    fname = "/home/phd/Documents/Journal/crawl"
    V, WE = loadEmbeddings(fname, corpusVoc)

    global N = 1
    global hidden_dim = size(WE, 1)
    global head_num = 4
    global head_dim = 300
    global ffn_dim = 256
    global bsize = 128

    @info "N : $(N), Hidden Dim : $(hidden_dim), Head Num: $(head_num), Head Dim: $(head_dim), FFN_dim: $(ffn_dim), BSize: $(bsize)"
    flush(io)
    global textenc, WE =  createTextEncoder(V, WE)
    global word_embeds = Embed(WE) |> todevice
    global pos_embed = SinCosPositionEmbed(hidden_dim)
    global encoder = Transformer(TransformerBlock, N, head_num, hidden_dim,
                          head_dim, ffn_dim) |> todevice

    global M = ENC(word_embeds, encoder)
    global ps = Flux.params(M.WEMBEDS, M.ENCODER)

    global opt = ADAM(λ)
    @info "Optimizer : ADAM with $(λ)"
    @warn "Starting Training for  $(dist)"
    train!(epochs=20, distance=dist)
    finalstamp = Dates.format(now(), "HH-MM-SS")
    @info "Finishing Job at :  $(finalstamp)"
end

close(io)

