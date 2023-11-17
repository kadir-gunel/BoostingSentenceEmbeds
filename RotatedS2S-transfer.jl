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
using Parameters


CUDA.device!(0) # setting for RTX 4500-(1) or 6000-(0)
enable_gpu(CUDA.functional()) # make `todevice` work on gpu if available


experiment = :seq2seqRotated
data = "wmt-crawl-ft"
ups = :wups


timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
save_path = "./models/sentences/$(experiment)/$(data)/$(ups)/$(timestamp)";
!isdir(save_path) ? mkpath(save_path) : nothing
logfile = "$(save_path)/log.$(ups).$(timestamp).txt"

README = "THIS IS THE EXPERIMENT WHERE A SEQ2SEQ MODEL IS BEING USED. THE MODEL CONSISTS OF :
         1. WORD EMBEDDING LAYER
         2. A ENCODER FOLLOWED BY A REDUCER: reduces sBert (768) to 300.
         The outputs of this reducer and the encoder are averaged.
         3. A DECODER: accepts the averaged results. Starts decoding
         process for PP minimization. At the end of this layer there is
         a augmentor network which is a simple 1 layered FFNN with elu
         activation function that augments the 300 outputs from decoder
         to 768 dimensional vectors.
         Objective function : PP + COSINE
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
    t = M.ENCODER(e, attention_mask)
    return t.hidden_state
end

function decoder_forward(input, m)
    attention_mask = get(input, :attention_mask, nothing)
    cross_attention_mask = get(input, :cross_attention_mask, nothing)
    e = embedding(input)
    t = M.DECODER(e,m, attention_mask, cross_attention_mask) # this includes causal_attention or masked_self_attention
    p = M.WDEMBEDS(t.hidden_state)
    dim, lengths, bsize = t.hidden_state |> size
    h = reshape(mean(t.hidden_state, dims=2), dim, bsize)
    return p, h
end

p_norm(M::T; dim=2) where {T} = sqrt.(sum(real(M .* conj(M)), dims=dim))
cosine_sim(X::T, Y::T) where {T} = diag((X ./ p_norm(X)) * (Y ./ p_norm(Y))')

cosine(X::T, Y::T) where {T} = 1 .- abs.(cosine_sim(X |> permutedims, Y |> permutedims))
euclidean(X::T, Y::T) where {T} = sqrt.(sum((X .- Y) .^ 2, dims=1))


function shift_decode_loss(logits, trg, trg_mask)
    label = trg[:, 2:end, :]
    return logitcrossentropy(@view(logits[:, 1:end-1, :]), label, trg_mask - 1)
end



function xloss(input)
    enc = encoder_forward(input.encoder_input)
    logits, _ = decoder_forward(input.decoder_input, enc)
    ce_loss =  shift_decode_loss(logits, input.decoder_input.token, input.decoder_input.attention_mask)
    return ce_loss
end 

function dloss(input, sbert; distance=euclidean)
    enc = encoder_forward(input.encoder_input)
    _, sent_vec = decoder_forward(input.decoder_input, enc)
    return mean(distance(sent_vec, sbert)) 
end



function combinedloss(input, sbert; distance=euclidean, α::Real=0.5, β::Real=0.5)
    enc = encoder_forward(input.encoder_input)
    logits, sent_vec = decoder_forward(input.decoder_input, enc)
    xloss =  shift_decode_loss(logits, input.decoder_input.token,
                                 input.decoder_input.attention_mask)
    dloss = mean(distance(sent_vec, sbert))
    return (α * xloss) + (β * dloss)
end


anynan(x) = any(y -> any(isnan, y), x)

function train!(; distance=:euclidean, epochs::Int=10)

    distance = (distance == :euclidean) || (distance == :cosine) ?
        eval(distance) : :xe
    
    loss_min = typemax(Float32)
    last_improvement = 0
    trn_size = floor(length(sentences) * 0.95) |> Int

    train_data = DataLoader((sentences[1:trn_size], sBertRed[:, 1:trn_size]), batchsize=bsize,
                            partial=false, shuffle=false)
    test_data = DataLoader((sentences[1+trn_size:end], sBertRed[:, 1+trn_size:end]), batchsize=bsize, partial=false, shuffle=false)

    # train_data = DataLoader((sentences[1:1000], sBertRed[:, 1:1000]), batchsize=bsize,
    #                        partial=false, shuffle=false)
    # test_data = DataLoader((sentences[1+1000:2000], sBertRed[:, 1+1000:2000]), batchsize=bsize,
    #                        partial=false, shuffle=false)

    for epoch in 1:epochs
        trn_losses = Float32[]; tst_losses = Float32[];
#         dgrad = nothing
        for(gsents, sbert) in train_data
            input = encode(textenc, gsents, gsents) |> todevice            

            cgrad = gradient(() -> combinedloss(input, sbert |>
                todevice, distance=distance), ps)

            #dgrad = !isequal(distance, :xe) ? gradient(()->
            #dloss(input, sbert |> todevice, distance=distance), ps) : nothing

            push!(trn_losses, combinedloss(input, sbert |> todevice, distance=distance))
            push!(trn_losses, dloss(input, sbert |> todevice, distance=euclidean))
            push!(trn_losses, dloss(input, sbert |> todevice, distance=cosine))
            push!(trn_losses, xloss(input)) # calculates xe loss

            
            # batch update!
            update!(opt, ps, cgrad)
            # !isequal(dgrad, nothing) ? update!(opt, ps, dgrad) : nothing
        end

        com = mean(trn_losses[1:4:end])
        euc = mean(trn_losses[2:4:end])
        cos = mean(trn_losses[3:4:end])
        xe  = mean(trn_losses[4:4:end])

        
        @info "Epoch: $(epoch) Combined Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos) XE Loss: $(xe) PP: $(exp(xe))"
        printstyled("Epoch: $(epoch)\n", color=:red)
        printstyled("Com Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos), XE Loss: $(xe), PPL $(exp(xe)) \n", color=:light_cyan)


        if anynan(ps)
           @error "NaN params"
           break
        end


        for(gsents, sbert) in test_data
            input = encode(textenc, gsents, gsents) |> todevice
            push!(tst_losses, combinedloss(input, sbert |> todevice, distance=distance))
            push!(tst_losses, dloss(input, sbert |> todevice, distance=euclidean))
            push!(tst_losses, dloss(input, sbert |> todevice, distance=cosine))
            push!(tst_losses, xloss(input)) # calculates xe loss
        end

        com = mean(tst_losses[1:4:end])
        euc = mean(tst_losses[2:4:end])
        cos = mean(tst_losses[3:4:end])
        xe  = mean(tst_losses[4:4:end])
        
        @info "Test Losses: Com Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos) XE Loss: $(xe) PP: $(exp(xe))"
        printstyled("Com Loss: $(com) Euc Loss: $(euc) Cos Loss: $(cos), XE Loss: $(xe), PPL $(exp(xe)) \n", color=:light_cyan)

        # if abs(mean(trn_losses) - loss_min) > 0.04
        M_CPU = S2S(M.WEMBEDS |> cpu, M.ENCODER |> cpu,  M.WDEMBEDS |> cpu, M.DECODER |> cpu)
            # model = Chain(word_embeds |> cpu, encoder |> cpu, reducer |> cpu,  decoder |> cpu, augmentor |> cpu) |> cpu
            # model = Flux.state(model)
        @save "$(save_path)/$(string(distance))/$(epoch).bson" M_CPU
        #    loss_min = mean(trn_losses[3:3:end]) # x-entropy
        #    last_improvement = epoch
        # end
        # w/o early stopping

#        if epoch - last_improvement >= 5 && opt.eta > 1e-6
#           opt.eta /= 10.0
#           @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
           # After dropping learning rate, give it a few epochs to improve
#           last_improvement = epoch
#        end

#         if epoch - last_improvement >= 10
#           @warn(" -> We're calling this converged.")
#           break
#        end
        


        flush(io) # writes to log file 
        
    end
end



flush(io) # writes all information till here  to log file 


@info "creating S2S model Object"
struct S2S
    WEMBEDS::Embed
    ENCODER::Transformer
    WDEMBEDS::EmbedDecoder
    DECODER::Transformer
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

# global λ = 1e-5
# global λ = 1e-3
global λ = 3e-4

# all losses include cross entropy
trn_list = [:cosine] #  :euclidean]
# trn_list = [:cosine, :xe]
for dist in trn_list
    mkdir(save_path * "/$(dist)")
    fname = "/home/phd/Documents/Journal/crawl"
    # fname = "./models/FT/$(data)"
    V, WE = loadEmbeddings(fname, corpusVoc)

    global N = 1
    global hidden_dim = size(WE, 1)
    global head_num = 4
    global head_dim = 300
    global ffn_dim = 256
    global bsize = 32
    
    @info "N_ENC : $(N), N_DEC : $(N), Hidden Dim : $(hidden_dim), Head Num: $(head_num), Head Dim: $(head_dim), FFN_dim: $(ffn_dim), BSize: $(bsize)"
    
    global textenc, WE =  createTextEncoder(V, WE)
    global word_embeds = Embed(WE) |> todevice
    global pos_embed = SinCosPositionEmbed(hidden_dim)
    global encoder = Transformer(TransformerBlock, N, head_num, hidden_dim, head_dim, ffn_dim) |> todevice
    global decoder = Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim) |> todevice
    global embed_decode = EmbedDecoder(word_embeds) # shared

    # creating model as S2S
    global M = S2S(word_embeds, encoder, embed_decode, decoder)

    global opt = ADAM(λ)
    global ps = Flux.params(M.WEMBEDS, M.ENCODER, M.DECODER)


    @info "Optimizer ADAM with $(λ)"
    @warn "Starting Training for  $(dist)"
    train!(epochs=40, distance=dist)
    finalstamp = Dates.format(now(), "HH-MM-SS")
    @info "Finishing Job at :  $(finalstamp)"

end

close(io)




