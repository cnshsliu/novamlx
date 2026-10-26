// Rendered from recursive-oss/rec `engine/src/rec_mlx/kernels/megakernel.py`
// (blob d833fac20e8d21bcca461116dd14e8c2bbf2e04b) through the same host
// pipeline `build_decoder` uses for official INT4 (`routed_expert_bits` 4,
// so the 3-bit mixed-expert specialization does not apply):
// expert teams → resident optimizations → rank routing → KV teams →
// QAT int8 embedding / bf16 residual rounding → fast routed activation.
//
// The 36/32/32 render matches the shader previously inlined in
// `FlintRecKernel` before the scratch aliases below. The 76/64/64 render is
// that pipeline with the M4 Max / M5 Max resident pool (rec device profile
// and the M4 Max measurements in recursive-oss/rec#17).
//
// Scratch aliases: workspace, partials, statistics, routing, and trace are
// session inputs. The greedy packet also keeps logits and the control buffer
// on the session. `release_control` zeros that buffer after the packet so
// the next launch does not upload zeros. Single-token decode still uploads
// a fresh control buffer and returns logits as an output. MLX marks inputs
// const; the kernel const-casts them the same way as KV.
enum FlintRecKernelSources {
    static let headerPro: String = #"""

struct KVView {
    const device bfloat16_t *p0, *p1, *p2, *p3;
    uint offset, cap;
    bool global;
    KVView head(uint index, uint dim) const {
        KVView view=*this; view.offset+=index*cap*128+dim; return view;
    }
    KVView value() const {
        KVView view=*this; view.offset+=cap*512; return view;
    }
    const device bfloat16_t* address(uint i) const {
        uint block=global?(i/128)>>16:0;
        uint local=global?i%(65536u*128u):i;
        const device bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;
        return p+offset+local;
    }
    bfloat16_t operator[](uint i) const { return *address(i); }
    void store(uint head, uint position, uint dim, bfloat16_t value) const {
        *((device bfloat16_t*)this->head(head,dim).address(position*128))=value;
    }
};

inline bool sync_all(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,device uint* profile,uint group) {

    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*36;
        uint arrived=atomic_fetch_add_explicit(control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        if(0) {uint at=(phase*36+group)*2;profile[at]=spins;profile[at+1]=arrived-phase*36;}
        ready[0]=uint(atomic_load_explicit(control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;

}

struct Matrix {
    const device uchar* w;
    const device bfloat16_t* s;
    const device bfloat16_t* b;
    uint n, k;
};
inline Matrix weight_matrix(const device uchar* bank, const device uint* d) {
    return {bank+d[0], (const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}
inline __attribute__((always_inline)) float load16(const device bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    // 8-byte tiles. Callers pass a 16-element cursor, so each tile is aligned.
    // One bf16->f32 conversion feeds both the sum and the shifted lane.
    for(uint i=0;i<16;i+=4) {
        vec<bfloat16_t,4> packed=*((const device vec<bfloat16_t,4>*)(x+i));
        float4 f=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
        sum+=f[0]+f[1]+f[2]+f[3];
        v[i]=f[0];v[i+1]=f[1]/16.0f;
        v[i+2]=f[2]/256.0f;v[i+3]=f[3]/4096.0f;
    }
    return sum;
}
inline __attribute__((always_inline)) float load16_local(const threadgroup bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        vec<bfloat16_t,4> packed=*((const threadgroup vec<bfloat16_t,4>*)(x+i));
        float4 f=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
        sum+=f[0]+f[1]+f[2]+f[3];
        v[i]=f[0];v[i+1]=f[1]/16.0f;
        v[i+2]=f[2]/256.0f;v[i+3]=f[3]/4096.0f;
    }
    return sum;
}
// Unshifted nibble masks, same left-associative sum as the ushort loop.
// `nibble_as_float` is exact for these masks (0..61440), so the dot matches
// `float(mask)` bit for bit. Banks are 256-byte aligned and row strides are
// multiples of 32 bytes, so the uint2 load is 8-byte aligned.
inline __attribute__((always_inline)) float nibble_as_float(uint bits) {
    return as_type<float>(0x4b000000u|bits)-8388608.0f;
}
inline __attribute__((always_inline)) float dot16(const device uchar* w,thread float* x,float s,float b,float sum) {
    uint2 packed=*((const device uint2*)w);
    float a=0.0f;
    uint word0=packed.x, word1=packed.y;
    a+=x[0]*nibble_as_float(word0&15u)+x[1]*nibble_as_float(word0&240u)
       +x[2]*nibble_as_float(word0&3840u)+x[3]*nibble_as_float(word0&61440u);
    a+=x[4]*nibble_as_float((word0>>16)&15u)+x[5]*nibble_as_float((word0>>16)&240u)
       +x[6]*nibble_as_float((word0>>16)&3840u)+x[7]*nibble_as_float((word0>>16)&61440u);
    a+=x[8]*nibble_as_float(word1&15u)+x[9]*nibble_as_float(word1&240u)
       +x[10]*nibble_as_float(word1&3840u)+x[11]*nibble_as_float(word1&61440u);
    a+=x[12]*nibble_as_float((word1>>16)&15u)+x[13]*nibble_as_float((word1>>16)&240u)
       +x[14]*nibble_as_float((word1>>16)&3840u)+x[15]*nibble_as_float((word1>>16)&61440u);
    return s*a+sum*b;
}
template<uint K>
inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups,uint expert=0) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
template<uint K>
inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups,uint expert=0) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16_local(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
inline void rms(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {
    float squared=0.0f;
    for(uint j=0;j<4;++j){float v=float(x[j]);squared+=v*v;}
    float sum=simd_sum(squared);
    if(sg==0)temp[lane]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0)temp[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){float total=simd_sum(temp[lane]);if(lane==0)temp[32]=metal::precise::rsqrt(total/2048.0f+eps);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<4;++j){uint i=tid*4+j;out[i]=w[i]*bfloat16_t(float(x[j])*temp[32]);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void head_norm(device bfloat16_t* x,const device bfloat16_t* w,uint heads,
                      bool rotary,uint position,float logbase,float eps,uint lane,uint sg) {
    for(uint h=sg;h<heads;h+=288) {
        device bfloat16_t* p=x+h*128;
        float squared=0.0f;for(uint j=0;j<4;++j){float v=p[lane*4+j];squared+=v*v;}
        float inverse=metal::precise::rsqrt(simd_sum(squared)/128.0f+eps);
        if(rotary) {
            for(uint j=0;j<2;++j) {
                uint pair=lane*2+j;
                bfloat16_t a=w[pair]*bfloat16_t(float(p[pair])*inverse);
                bfloat16_t b=w[pair+64]*bfloat16_t(float(p[pair+64])*inverse);
                float theta=float(position)*metal::exp2(-float(pair)/64.0f*logbase);
                float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
                p[pair]=bfloat16_t(float(a)*c-float(b)*s);
                p[pair+64]=bfloat16_t(float(a)*s+float(b)*c);
            }
        } else for(uint j=0;j<4;++j){uint d=lane*4+j;p[d]=w[d]*bfloat16_t(float(p[d])*inverse);}
    }
}
inline uint partial_offset(uint head,uint part,uint dim,uint parts,bool rounded) {
    return rounded?(head*32+dim/4)*parts*4+part*4+dim%4:(head*parts+part)*128+dim;
}
// Exact immutable layer metadata, cooperatively copied once per group.
inline Matrix weight_matrix(const device uchar* bank,const threadgroup uint* d) {
    return {bank+d[0],(const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}

template<bool ROUNDED>
inline void attention_merge_vec4_w2(device bfloat16_t* out,
                      device float* partial,device float* stats,
                      uint parts,uint lane,uint sg) {
    // A SIMD owns 16 output dimensions. Every scalar retains its original
    // per-lane partition order and simd_sum; loads follow native four-wide tiles.
    for(uint task=(sg/72)*64+sg%72;task<(sg/72+1)*64;task+=72) {
        uint h=task/8,dim=(task%8)*16;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);
        float z=0.0f;
        float4 a[4]={float4(0),float4(0),float4(0),float4(0)};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;
            float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint tile=0;tile<4;++tile) {
                float4 value;
                if constexpr(ROUNDED) {
                    uint at=((h*32+dim/4+tile)*parts+p)*4;
                    vec<bfloat16_t,4> raw=*((const device vec<bfloat16_t,4>*)(((const device bfloat16_t*)partial)+at));
                    value=float4(float(raw[0]),float(raw[1]),float(raw[2]),float(raw[3]));
                } else {
                    uint at=(h*parts+p)*128+dim+tile*4;
                    value=*((const device float4*)(partial+at));
                }
                a[tile]+=factor*value;
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint tile=0;tile<4;++tile)for(uint component=0;component<4;++component) {
            uint j=tile*4+component;
            float total=simd_sum(a[tile][component]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<8)for(uint t=0;t<2;++t) {
            uint j=lane*2+t;
            out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);
        }
    }
}

template<uint HEADS>
inline void attention_partial(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<(32/HEADS)*parts;task+=288) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(q[(first+h)*128+lane*4+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(k[at+j]);value[j]=float(v[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4+j;
            next_key[j]=k[at];next_value[j]=v[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                size_t next=at+size_t(parts)*128+j;
                next_key[j]=k[next];next_value[j]=v[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS){uint row=(first+lane)*parts+part;stats[row*2]=hi;stats[row*2+1]=z;}
        for(uint h=0;h<HEADS;++h) {
            for(uint j=0;j<4;++j) {
                uint at=partial_offset(first+h,part,lane*4+j,parts,rounded);
                // Global partitions already round here in the native implementation.
                if(rounded)((device bfloat16_t*)partial)[at]=bfloat16_t(acc[h][j]);
                else partial[at]=acc[h][j];
            }
        }
    }
}
inline void attention_merge(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<32*16;task+=288) {
        uint h=task/16,dim=(task%16)*8;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);float z=0.0f,a[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint j=0;j<8;++j) {
                uint at=partial_offset(h,p,dim+j,parts,rounded);
                a[j]+=factor*(rounded?float(((device bfloat16_t*)partial)[at]):partial[at]);
            }
        }
        z=simd_sum(z);
        for(uint j=0;j<8;++j){float total=simd_sum(a[j]);if(lane==0)out[h*128+dim+j]=bfloat16_t(z==0.0f?total:total/z);}
    }
}
inline void route_dot_bf16(const threadgroup bfloat16_t* x,const device bfloat16_t* w,
                      device float* probs,uint lane,uint sg) {
    for(uint row=sg;row<256;row+=288) {
        float a=0.0f;
        for(uint b=0;b<16;++b) {
            uint at=b*128+lane*4;
            const vec<bfloat16_t,4> packed=*((const device vec<bfloat16_t,4>*)(w+row*2048+at));
            const float4 weights=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
            for(uint j=0;j<4;++j)a+=weights[j]*float(x[at+j]);
        }
        for(ushort d=16;d>=1;d>>=1)a+=simd_shuffle_down(a,d);
        a=float(bfloat16_t(a));
        if(lane==0){float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(a)));probs[row]=a<0?tail:1.0f-tail;}
    }
}
inline void route_top(const device float* bias,device float* probs,
                      device uint* ids,device float* scores,uint lane,uint sg) {
    if(sg==0) {
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<6;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i>idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_max(best==peak?idx:0);
            float value=simd_shuffle(p[selected/32],selected%32);
            if(lane==0){ids[5-rank]=selected;scores[5-rank]=value;}
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
        if(lane==0){float sum=0.0f;for(uint j=0;j<6;++j)sum=scores[j]+sum;sum+=1.0e-20f;for(uint j=0;j<6;++j)scores[j]/=sum;}
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline void gateup_all(const device uchar* bank,const threadgroup uint* d,
                      const threadgroup bfloat16_t* x,const device uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<4096;first+=256*4) {
        bool shared=first>=3072;uint role=first/512;
        uint local=shared?first-3072:first%512;
        Matrix g=weight_matrix(bank,d+(shared?45:30));
        Matrix u=weight_matrix(bank,d+(shared?50:35));
        uint expert=shared?0:ids[role];
        float ga[4]={0.0f,0.0f,0.0f,0.0f},ua[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[first+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[first+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
inline void down_all(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<7*2048;first+=256*4) {
        uint role=first/2048,local=first%2048;bool shared=role==6;
        Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
        const device bfloat16_t* input=x+role*512;
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[first+j]=bfloat16_t(v);}
    }
}


template<uint HEADS>
inline void attention_partial_w204(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<(32/HEADS)*parts;task+=288) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        const device bfloat16_t* key_base=k+size_t(kv)*cap*128+lane*4;
        const device bfloat16_t* value_base=v+size_t(kv)*cap*128+lane*4;
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            uint at=uint(token)*128;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            uint at=uint(token)*128+j;
            next_key[j]=key_base[at];next_value[j]=value_base[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            uint at=uint(token)*128;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                uint next=at+parts*128+j;
                next_key[j]=key_base[next];next_value[j]=value_base[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS) {
            device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;
            stat_out[0]=hi;stat_out[1]=z;
        }
        for(uint h=0;h<HEADS;++h) {
            device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4:(device bfloat16_t*)partial;
            device float* unrounded_out=rounded?partial:partial+((first+h)*parts+part)*128+lane*4;
            if(rounded) {
                for(uint j=0;j<4;j+=4)
                    *((device vec<bfloat16_t,4>*)(rounded_out+j))=vec<bfloat16_t,4>(bfloat16_t(acc[h][j+0]),bfloat16_t(acc[h][j+1]),bfloat16_t(acc[h][j+2]),bfloat16_t(acc[h][j+3]));
            } else {
                for(uint j=0;j<4;j+=4)
                    *((device float4*)(unrounded_out+j))=float4(acc[h][j+0],acc[h][j+1],acc[h][j+2],acc[h][j+3]);
            }
        }
    }
}

inline __attribute__((always_inline)) float dot16_w197_bias23(const device uchar* w,thread float* x,float s,float b,float sum) {
    return dot16(w,x,s,b,sum);
}

inline void down_all_w197_bias23(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<7*2048;first+=256*4) {
        uint role=first/2048,local=first%2048;bool shared=role==6;
        Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
        const device bfloat16_t* input=x+role*512;
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[first+j]=bfloat16_t(v);}
    }
}

template<uint WIDTH>
inline void attention_merge_w182(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<32*(128/WIDTH);task+=288) {
        uint h=task/(128/WIDTH),dim=(task%(128/WIDTH))*WIDTH;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);float z=0.0f,a[WIDTH]={};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint j=0;j<WIDTH;++j) {
                uint at=partial_offset(h,p,dim+j,parts,rounded);
                a[j]+=factor*(rounded?float(((device bfloat16_t*)partial)[at]):partial[at]);
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint j=0;j<WIDTH;++j) {
            float total=simd_sum(a[j]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<WIDTH/2)for(uint t=0;t<2;++t) {
            uint j=lane*2+t;
            out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);
        }
    }
}

template<uint HEADS,bool ROUNDED>
inline void attention_partial_path_w53(const device bfloat16_t* q,KVView k,
                      KVView v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool runtime_rounded,uint lane,uint sg) {
    constexpr bool rounded=ROUNDED;
    for(uint task=(sg/72)*((8/HEADS)*parts)+sg%72;task<(sg/72+1)*((8/HEADS)*parts);task+=72) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*4);
        KVView value_base=v.head(kv,lane*4);
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            uint at=uint(token)*128;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            uint at=uint(token)*128+j;
            next_key[j]=key_base[at];next_value[j]=value_base[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            uint at=uint(token)*128;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                uint next=at+parts*128+j;
                next_key[j]=key_base[next];next_value[j]=value_base[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS) {
            device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;
            stat_out[0]=hi;stat_out[1]=z;
        }
        for(uint h=0;h<HEADS;++h) {
            device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4:(device bfloat16_t*)partial;
            device float* unrounded_out=rounded?partial:partial+((first+h)*parts+part)*128+lane*4;
            if(rounded) {
                for(uint j=0;j<4;j+=4)
                    *((device vec<bfloat16_t,4>*)(rounded_out+j))=vec<bfloat16_t,4>(bfloat16_t(acc[h][j+0]),bfloat16_t(acc[h][j+1]),bfloat16_t(acc[h][j+2]),bfloat16_t(acc[h][j+3]));
            } else {
                for(uint j=0;j<4;j+=4)
                    *((device float4*)(unrounded_out+j))=float4(acc[h][j+0],acc[h][j+1],acc[h][j+2],acc[h][j+3]);
            }
        }
    }
}


template<bool LANE_SCORES>
inline void route_top_lane_scores(const device float* bias,device float* probs,
                      device uint* ids,device float* scores,uint lane,uint sg) {
    if(sg==0) {
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        float chosen=0.0f;
        for(uint rank=0;rank<6;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i>idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_max(best==peak?idx:0);
            float value=simd_shuffle(p[selected/32],selected%32);
            if(lane==0)ids[5-rank]=selected;
            if constexpr(LANE_SCORES) {
                if(lane==5-rank)chosen=value;
            } else {
                if(lane==0)scores[5-rank]=value;
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
        if constexpr(LANE_SCORES) {
            float sum=0.0f;
            for(uint j=0;j<6;++j)sum=simd_shuffle(chosen,j)+sum;
            sum+=1.0e-20f;
            if(lane<6)scores[lane]=chosen/sum;
        } else {
            if(lane==0){float sum=0.0f;for(uint j=0;j<6;++j)sum=scores[j]+sum;sum+=1.0e-20f;for(uint j=0;j<6;++j)scores[j]/=sum;}
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}
inline void rms_256_w4(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {

    float sums[2];
    for(uint leaf_set=0;leaf_set<2;++leaf_set) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=float(x[leaf_set*4+j]);squared+=v*v;}
        sums[leaf_set]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+eps);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=w[i]*bfloat16_t(float(x[j])*inverse);}
    threadgroup_barrier(mem_flags::mem_threadgroup);

}

// Independent expert teams: two groups per routed role, four for shared.
inline void expert_team_gateup(const device uchar* bank,const threadgroup uint* d,
                      const threadgroup bfloat16_t* x,const device uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,
                      uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4, team=shared?group-24:group%4;
    uint teams=shared?8:4, width=shared?1024:512;
    Matrix g=weight_matrix(bank,d+(shared?45:30));
    Matrix u=weight_matrix(bank,d+(shared?50:35));
    uint expert=shared?0:ids[role];
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*2;local<width;local+=teams*8*2) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[role*512+local+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::fast::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[role*512+local+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
inline void expert_team_down(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint local_sg,uint group) {
    bool shared=group>=24;
    uint role=shared?6:group/4, team=shared?group-24:group%4;
    uint teams=shared?8:4;
    Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
    const device bfloat16_t* input=x+role*512;
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*4;local<2048;local+=teams*8*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(v);}
    }
}

inline bool expert_team_handoff(device uint* routing,device atomic_uint* control,
                    uint tid,uint group,uint layer,uint phase,threadgroup uint* ready) {
    uint first=group>=24?24:(group/4)*4, count=group>=24?8:4;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),layer+1,memory_order_relaxed);
        bool complete=true;
        for(uint peer=first;peer<first+count;++peer) {
            uint spins=0;
            while(atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)<layer+1 &&
                  atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
            complete=complete && atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)>=layer+1;
        }
        ready[0]=uint(complete && atomic_load_explicit(control+2,memory_order_relaxed)==0);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    return ready[0]!=0;
}

inline void w3_route_own_rank(const device float* bias,device float* probs,
                            device float* chosen,device uint* ids,uint lane,uint sg,uint group) {
    if(sg==0 && group<24) {
        uint role=group/4;
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<=role;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            if(rank==role) {
                float value=simd_shuffle(p[selected/32],selected%32);
                if(lane==0) {
                    ids[role]=selected;
                    if(group%4==0)chosen[role]=value;
                }
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline bool kv_team_sync(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,device uint* profile,uint group) {
    device atomic_uint* team_control=control+4+(group/9)*2;

    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*9;
        uint arrived=atomic_fetch_add_explicit(team_control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(team_control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        if(0) {uint at=(phase*36+group)*2;profile[at]=spins;profile[at+1]=arrived-phase*36;}
        ready[0]=uint(atomic_load_explicit(team_control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;

}


template<uint K>
inline void q4_local_block(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint first) {
    float a[4]={0.0f,0.0f,0.0f,0.0f};
    for(uint block=0;block<K;block+=512) {
        uint scalar=block+lane*16;
        float v[16];float sum=load16_local(x+scalar,v);
        for(uint r=0;r<4;++r) {
            uint row=first+r;uint gi=row*(K/64)+scalar/64;
            a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
        }
    }
    for(uint r=0;r<4;++r) {
        float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
    }
}

// Last threadgroup to leave the launch zeros the spin counters in place.
// Greedy packets reuse this buffer. The next launch must observe zeros or
// sync_all waits forever. Words 0..63 cover the grid counter, the failure
// flag, KV team counters, and the pro ticket counters. Word 1022 is the
// arrival count. Single-token decode does not call this.
inline void release_control(device atomic_uint* control, uint tid) {
    if(tid==0) {
        uint n=atomic_fetch_add_explicit(control+1022,1u,memory_order_relaxed);
        if(n+1u==36u) {
            for(uint i=0;i<64u;++i) atomic_store_explicit(control+i,0u,memory_order_relaxed);
            atomic_store_explicit(control+1022,0u,memory_order_relaxed);
        }
    }
}


"""#
    static let bodyPro: String = #"""
#define KV_TEAM_SYNC() if(!kv_team_sync(control,tid,kv_phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}

    uint token=parameters[0], position=parameters[1], initial_position=parameters[2], layer_count=parameters[3];
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,sg=group*8+simdgroup_index_in_threadgroup;
    uint index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    device bfloat16_t* trace_rw=(device bfloat16_t*)trace;
    device bfloat16_t* workspace_rw=(device bfloat16_t*)workspace;
    device float* partials_rw=(device float*)partials;
    device float* statistics_rw=(device float*)statistics;
    device uint* routing_rw=(device uint*)routing;
    #define trace trace_rw
    #define workspace workspace_rw
    #define partials partials_rw
    #define statistics statistics_rw
    #define routing routing_rw
    threadgroup uint ready[1];uint phase=0;uint kv_phase=0;uint kv_qkv_base=0;
    #define SYNC() if(!sync_all(control,tid,phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}


    threadgroup float reduction[33];
    threadgroup uint layer_descriptor[64];
    // Every resident group consumes scores in its private residual update.
    device uint* ids=routing+group*16;
    device float* scores=(device float*)(routing+group*16+8);
    if(tid==0)atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    bfloat16_t h[8];
    alignas(8) threadgroup bfloat16_t n[2048];
    device bfloat16_t* q=workspace+4096;
    device bfloat16_t* k=workspace+8192;
    device bfloat16_t* v=workspace+8704;
    device bfloat16_t* a=workspace+9216;
    device bfloat16_t* o=workspace+13312;
    device bfloat16_t* gate=workspace+15360;
    device bfloat16_t* up=workspace+31744;
    device bfloat16_t* act=workspace+48128;
    device bfloat16_t* down=workspace+64512;
    const device uint* final=metadata+49*64;
    const device uchar* tail=bank12;
    Matrix embedding=weight_matrix(tail,final+10);
    for(uint j=0;j<8;++j) {
        uint i=(tid+(j/4)*256)*4+j%4;
        size_t row=uint(token);uint code=embedding.w[row*2048+i];
        uint gi=row*32+i/64;
        h[j]=bfloat16_t(embedding.s[gi]*bfloat16_t(code)+embedding.b[gi]);
        if(group==0)trace[i]=h[j];
    }
    SYNC();
    for(uint layer=0;layer<uint(layer_count);++layer) {
        if(tid<64)layer_descriptor[tid]=metadata[layer*64+tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup uint* d=layer_descriptor;
        const device uchar* bank=layer<4?bank0:layer<8?bank1:layer<12?bank2:layer<16?bank3:layer<20?bank4:layer<24?bank5:layer<28?bank6:layer<32?bank7:layer<36?bank8:layer<40?bank9:layer<44?bank10:layer<48?bank11:bank12;
        bool global=d[1]!=0,sparse=d[2]!=0;
        {rms_256_w4(h,(const device bfloat16_t*)(bank+d[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
        uint cap=d[62],count=global?uint(position)+1:min(uint(position)+1,cap);
        uint insert=global?uint(position):(uint(initial_position)>=cap?(uint(position)-uint(initial_position))%cap:uint(position)%cap);
        KVView keys={kv0,kv1,kv2,kv3,d[61],cap,global};
        KVView vals=keys.value();
        uint kv_team=group/9,kv_member=group%9;
        for(;;) {
            uint ticket=0;
            if(lane==0)ticket=atomic_fetch_add_explicit(control+40+kv_team,1u,memory_order_relaxed);
            ticket=simd_broadcast(ticket,0);
            if(ticket>=kv_qkv_base+320u)break;
            uint unit=ticket-kv_qkv_base,member=unit/32u;
            uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
            m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
            q4_local_block<2048>(m,n,(head<32?q:head<36?k:v)+local*128,lane,(unit%32u)*4u);
        }
        kv_qkv_base+=392u;
        KV_TEAM_SYNC();
        for(uint member=kv_member;member<10;member+=9) {
            uint head=member<8?kv_team*8+member:member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
            if(simdgroup_index_in_threadgroup==0 && head<36)
                head_norm(out,(const device bfloat16_t*)(bank+d[head<32?5:6]),1,!global,uint(position),13.287712379549449f,1e-06f,lane,0);
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {
                KVView target=head<36?keys:vals;
                for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
            }
        }
        KV_TEAM_SYNC();
        bool rounded=global&&count>=1024;
        uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
        if(group<36) {
            if(global && count>1024) {
                if(rounded) attention_partial_path_w53<8,true>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<8,false>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
            else {
                if(rounded) attention_partial_path_w53<4,true>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<4,false>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
        }
        KV_TEAM_SYNC();
        if(group<36) {if(rounded) attention_merge_vec4_w2<true>(a,partials,statistics,parts,lane,sg); else attention_merge_vec4_w2<false>(a,partials,statistics,parts,lane,sg);}
        SYNC();
        q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,288);
        SYNC();
        {
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=bfloat16_t(float(h[j])+float(o[i]));}
            rms_256_w4(h,(const device bfloat16_t*)(bank+d[4]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);
        }
        if(sparse) {
            if(group<36) route_dot_bf16(n,(const device bfloat16_t*)(bank+d[60]),statistics,lane,sg);
            SYNC();
            w3_route_own_rank((const device float*)(bank+d[9]),statistics,(device float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);
            // The local route_top device fence publishes this group's ids and scores.
            if(32==32) {
                if(group<32) {
                    expert_team_gateup(bank,d,n,ids,silu_table,act,lane,simdgroup_index_in_threadgroup,group);
                    if(!expert_team_handoff(routing,control,tid,group,layer,phase,ready)) {
                        if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);
                        return;
                    }
                    expert_team_down(bank,d,act,ids,down,lane,simdgroup_index_in_threadgroup,group);
                }
            } else {
                if(group<32) gateup_all(bank,d,n,ids,silu_table,act,lane,sg);
                SYNC();
                if(group<32) down_all_w197_bias23(bank,d,act,ids,down,lane,sg);
            }
            SYNC();
        } else {
            Matrix gm=weight_matrix(bank,d+30),um=weight_matrix(bank,d+35),dm=weight_matrix(bank,d+40);
            q4_local<2048>(gm,n,gate,lane,sg,288);q4_local<2048>(um,n,up,lane,sg,288);
            SYNC();
            for(uint i=index;i<gm.n;i+=9216) {
                bfloat16_t g=gate[i],u=up[i];
                auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;
                act[i]=bfloat16_t(bfloat16_t(g*sig)*u);
            }
            SYNC();
            q4<6144>(dm,act,down,lane,sg,288);
            SYNC();
        }
        {
        if(sparse) {
            float w3_chosen=lane<6?((device float*)(routing+8))[lane]:0.0f;
            float w3_sum=0.0f;
            for(uint j=0;j<6;++j)w3_sum=simd_shuffle(w3_chosen,j)+w3_sum;
            w3_sum+=1.0e-20f;
            float w3_weight=lane<6?w3_chosen/w3_sum:0.0f;
            float sums[8];
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t(down[i]*bfloat16_t(simd_shuffle(w3_weight,ushort(0)))))+0.0f;
            }
            for(uint j=0;j<8;++j)for(uint r=1;r<6;++r) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t((float(bfloat16_t(down[r*2048+i]*bfloat16_t(simd_shuffle(w3_weight,ushort(r)))))+0.0f)+sums[j]));
            }
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=bfloat16_t(float(bfloat16_t(sums[j]))+float(down[6*2048+i]));
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        } else {
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=down[i];
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        }
        }
    }
    {rms_256_w4(h,(const device bfloat16_t*)(tail+final[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
    if(group<32) q4_local<2048>(weight_matrix(tail,final+30),n,logits,lane,sg,256);
    if(tid==0)diagnostic[group]=0;
    if(0 && tid==0 && group==0)profile[512*36*2]=phase;

"""#
    static let headerMax: String = #"""

struct KVView {
    const device bfloat16_t *p0, *p1, *p2, *p3;
    uint offset, cap;
    bool global;
    KVView head(uint index, uint dim) const {
        KVView view=*this; view.offset+=index*cap*128+dim; return view;
    }
    KVView value() const {
        KVView view=*this; view.offset+=cap*512; return view;
    }
    const device bfloat16_t* address(uint i) const {
        uint block=global?(i/128)>>16:0;
        uint local=global?i%(65536u*128u):i;
        const device bfloat16_t* p=block==0?p0:block==1?p1:block==2?p2:p3;
        return p+offset+local;
    }
    bfloat16_t operator[](uint i) const { return *address(i); }
    void store(uint head, uint position, uint dim, bfloat16_t value) const {
        *((device bfloat16_t*)this->head(head,dim).address(position*128))=value;
    }
};

inline bool sync_all(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,device uint* profile,uint group) {

    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*76;
        uint arrived=atomic_fetch_add_explicit(control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        if(0) {uint at=(phase*76+group)*2;profile[at]=spins;profile[at+1]=arrived-phase*76;}
        ready[0]=uint(atomic_load_explicit(control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;

}

struct Matrix {
    const device uchar* w;
    const device bfloat16_t* s;
    const device bfloat16_t* b;
    uint n, k;
};
inline Matrix weight_matrix(const device uchar* bank, const device uint* d) {
    return {bank+d[0], (const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}
inline __attribute__((always_inline)) float load16(const device bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    // 8-byte tiles. Callers pass a 16-element cursor, so each tile is aligned.
    // One bf16->f32 conversion feeds both the sum and the shifted lane.
    for(uint i=0;i<16;i+=4) {
        vec<bfloat16_t,4> packed=*((const device vec<bfloat16_t,4>*)(x+i));
        float4 f=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
        sum+=f[0]+f[1]+f[2]+f[3];
        v[i]=f[0];v[i+1]=f[1]/16.0f;
        v[i+2]=f[2]/256.0f;v[i+3]=f[3]/4096.0f;
    }
    return sum;
}
inline __attribute__((always_inline)) float load16_local(const threadgroup bfloat16_t* x, thread float* v) {
    float sum=0.0f;
    for(uint i=0;i<16;i+=4) {
        vec<bfloat16_t,4> packed=*((const threadgroup vec<bfloat16_t,4>*)(x+i));
        float4 f=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
        sum+=f[0]+f[1]+f[2]+f[3];
        v[i]=f[0];v[i+1]=f[1]/16.0f;
        v[i+2]=f[2]/256.0f;v[i+3]=f[3]/4096.0f;
    }
    return sum;
}
// Unshifted nibble masks, same left-associative sum as the ushort loop.
// `nibble_as_float` is exact for these masks (0..61440), so the dot matches
// `float(mask)` bit for bit. Banks are 256-byte aligned and row strides are
// multiples of 32 bytes, so the uint2 load is 8-byte aligned.
inline __attribute__((always_inline)) float nibble_as_float(uint bits) {
    return as_type<float>(0x4b000000u|bits)-8388608.0f;
}
inline __attribute__((always_inline)) float dot16(const device uchar* w,thread float* x,float s,float b,float sum) {
    uint2 packed=*((const device uint2*)w);
    float a=0.0f;
    uint word0=packed.x, word1=packed.y;
    a+=x[0]*nibble_as_float(word0&15u)+x[1]*nibble_as_float(word0&240u)
       +x[2]*nibble_as_float(word0&3840u)+x[3]*nibble_as_float(word0&61440u);
    a+=x[4]*nibble_as_float((word0>>16)&15u)+x[5]*nibble_as_float((word0>>16)&240u)
       +x[6]*nibble_as_float((word0>>16)&3840u)+x[7]*nibble_as_float((word0>>16)&61440u);
    a+=x[8]*nibble_as_float(word1&15u)+x[9]*nibble_as_float(word1&240u)
       +x[10]*nibble_as_float(word1&3840u)+x[11]*nibble_as_float(word1&61440u);
    a+=x[12]*nibble_as_float((word1>>16)&15u)+x[13]*nibble_as_float((word1>>16)&240u)
       +x[14]*nibble_as_float((word1>>16)&3840u)+x[15]*nibble_as_float((word1>>16)&61440u);
    return s*a+sum*b;
}
template<uint K>
inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups,uint expert=0) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
template<uint K>
inline void q4_local(Matrix m,const threadgroup bfloat16_t* x,device bfloat16_t* out,
               uint lane,uint sg,uint groups,uint expert=0) {
    for(uint first=sg*4;first<m.n;first+=groups*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint block=0;block<K;block+=512) {
            uint scalar=block+lane*16;
            float v[16];float sum=load16_local(x+scalar,v);
            for(uint r=0;r<4;++r) {
                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;
                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint r=0;r<4;++r) {
            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);
        }
    }
}
inline void rms(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {
    float squared=0.0f;
    for(uint j=0;j<4;++j){float v=float(x[j]);squared+=v*v;}
    float sum=simd_sum(squared);
    if(sg==0)temp[lane]=0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0)temp[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0){float total=simd_sum(temp[lane]);if(lane==0)temp[32]=metal::precise::rsqrt(total/2048.0f+eps);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<4;++j){uint i=tid*4+j;out[i]=w[i]*bfloat16_t(float(x[j])*temp[32]);}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void head_norm(device bfloat16_t* x,const device bfloat16_t* w,uint heads,
                      bool rotary,uint position,float logbase,float eps,uint lane,uint sg) {
    for(uint h=sg;h<heads;h+=608) {
        device bfloat16_t* p=x+h*128;
        float squared=0.0f;for(uint j=0;j<4;++j){float v=p[lane*4+j];squared+=v*v;}
        float inverse=metal::precise::rsqrt(simd_sum(squared)/128.0f+eps);
        if(rotary) {
            for(uint j=0;j<2;++j) {
                uint pair=lane*2+j;
                bfloat16_t a=w[pair]*bfloat16_t(float(p[pair])*inverse);
                bfloat16_t b=w[pair+64]*bfloat16_t(float(p[pair+64])*inverse);
                float theta=float(position)*metal::exp2(-float(pair)/64.0f*logbase);
                float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
                p[pair]=bfloat16_t(float(a)*c-float(b)*s);
                p[pair+64]=bfloat16_t(float(a)*s+float(b)*c);
            }
        } else for(uint j=0;j<4;++j){uint d=lane*4+j;p[d]=w[d]*bfloat16_t(float(p[d])*inverse);}
    }
}
inline uint partial_offset(uint head,uint part,uint dim,uint parts,bool rounded) {
    return rounded?(head*32+dim/4)*parts*4+part*4+dim%4:(head*parts+part)*128+dim;
}
// Exact immutable layer metadata, cooperatively copied once per group.
inline Matrix weight_matrix(const device uchar* bank,const threadgroup uint* d) {
    return {bank+d[0],(const device bfloat16_t*)(bank+d[1]),
            (const device bfloat16_t*)(bank+d[2]),d[3],d[4]};
}

template<bool ROUNDED>
inline void attention_merge_vec4_w2(device bfloat16_t* out,
                      device float* partial,device float* stats,
                      uint parts,uint lane,uint sg) {
    // A SIMD owns 16 output dimensions. Every scalar retains its original
    // per-lane partition order and simd_sum; loads follow native four-wide tiles.
    for(uint task=(sg/152)*64+sg%152;task<(sg/152+1)*64;task+=152) {
        uint h=task/8,dim=(task%8)*16;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);
        float z=0.0f;
        float4 a[4]={float4(0),float4(0),float4(0),float4(0)};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;
            float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint tile=0;tile<4;++tile) {
                float4 value;
                if constexpr(ROUNDED) {
                    uint at=((h*32+dim/4+tile)*parts+p)*4;
                    vec<bfloat16_t,4> raw=*((const device vec<bfloat16_t,4>*)(((const device bfloat16_t*)partial)+at));
                    value=float4(float(raw[0]),float(raw[1]),float(raw[2]),float(raw[3]));
                } else {
                    uint at=(h*parts+p)*128+dim+tile*4;
                    value=*((const device float4*)(partial+at));
                }
                a[tile]+=factor*value;
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint tile=0;tile<4;++tile)for(uint component=0;component<4;++component) {
            uint j=tile*4+component;
            float total=simd_sum(a[tile][component]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<8)for(uint t=0;t<2;++t) {
            uint j=lane*2+t;
            out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);
        }
    }
}

template<uint HEADS>
inline void attention_partial(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<(32/HEADS)*parts;task+=608) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(q[(first+h)*128+lane*4+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(k[at+j]);value[j]=float(v[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4+j;
            next_key[j]=k[at];next_value[j]=v[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            size_t at=(size_t(kv)*cap+token)*128+lane*4;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                size_t next=at+size_t(parts)*128+j;
                next_key[j]=k[next];next_value[j]=v[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS){uint row=(first+lane)*parts+part;stats[row*2]=hi;stats[row*2+1]=z;}
        for(uint h=0;h<HEADS;++h) {
            for(uint j=0;j<4;++j) {
                uint at=partial_offset(first+h,part,lane*4+j,parts,rounded);
                // Global partitions already round here in the native implementation.
                if(rounded)((device bfloat16_t*)partial)[at]=bfloat16_t(acc[h][j]);
                else partial[at]=acc[h][j];
            }
        }
    }
}
inline void attention_merge(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<32*16;task+=608) {
        uint h=task/16,dim=(task%16)*8;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);float z=0.0f,a[8]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint j=0;j<8;++j) {
                uint at=partial_offset(h,p,dim+j,parts,rounded);
                a[j]+=factor*(rounded?float(((device bfloat16_t*)partial)[at]):partial[at]);
            }
        }
        z=simd_sum(z);
        for(uint j=0;j<8;++j){float total=simd_sum(a[j]);if(lane==0)out[h*128+dim+j]=bfloat16_t(z==0.0f?total:total/z);}
    }
}
inline void route_dot_bf16(const threadgroup bfloat16_t* x,const device bfloat16_t* w,
                      device float* probs,uint lane,uint sg) {
    for(uint row=sg;row<256;row+=608) {
        float a=0.0f;
        for(uint b=0;b<16;++b) {
            uint at=b*128+lane*4;
            const vec<bfloat16_t,4> packed=*((const device vec<bfloat16_t,4>*)(w+row*2048+at));
            const float4 weights=float4(float(packed[0]),float(packed[1]),float(packed[2]),float(packed[3]));
            for(uint j=0;j<4;++j)a+=weights[j]*float(x[at+j]);
        }
        for(ushort d=16;d>=1;d>>=1)a+=simd_shuffle_down(a,d);
        a=float(bfloat16_t(a));
        if(lane==0){float tail=1.0f/(1.0f+metal::precise::exp(metal::abs(a)));probs[row]=a<0?tail:1.0f-tail;}
    }
}
inline void route_top(const device float* bias,device float* probs,
                      device uint* ids,device float* scores,uint lane,uint sg) {
    if(sg==0) {
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<6;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i>idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_max(best==peak?idx:0);
            float value=simd_shuffle(p[selected/32],selected%32);
            if(lane==0){ids[5-rank]=selected;scores[5-rank]=value;}
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
        if(lane==0){float sum=0.0f;for(uint j=0;j<6;++j)sum=scores[j]+sum;sum+=1.0e-20f;for(uint j=0;j<6;++j)scores[j]/=sum;}
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline void gateup_all(const device uchar* bank,const threadgroup uint* d,
                      const threadgroup bfloat16_t* x,const device uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<4096;first+=512*4) {
        bool shared=first>=3072;uint role=first/512;
        uint local=shared?first-3072:first%512;
        Matrix g=weight_matrix(bank,d+(shared?45:30));
        Matrix u=weight_matrix(bank,d+(shared?50:35));
        uint expert=shared?0:ids[role];
        float ga[4]={0.0f,0.0f,0.0f,0.0f},ua[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[first+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[first+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
inline void down_all(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<7*2048;first+=512*4) {
        uint role=first/2048,local=first%2048;bool shared=role==6;
        Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
        const device bfloat16_t* input=x+role*512;
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[first+j]=bfloat16_t(v);}
    }
}


template<uint HEADS>
inline void attention_partial_w204(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<(32/HEADS)*parts;task+=608) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        const device bfloat16_t* key_base=k+size_t(kv)*cap*128+lane*4;
        const device bfloat16_t* value_base=v+size_t(kv)*cap*128+lane*4;
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            uint at=uint(token)*128;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            uint at=uint(token)*128+j;
            next_key[j]=key_base[at];next_value[j]=value_base[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            uint at=uint(token)*128;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                uint next=at+parts*128+j;
                next_key[j]=key_base[next];next_value[j]=value_base[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS) {
            device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;
            stat_out[0]=hi;stat_out[1]=z;
        }
        for(uint h=0;h<HEADS;++h) {
            device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4:(device bfloat16_t*)partial;
            device float* unrounded_out=rounded?partial:partial+((first+h)*parts+part)*128+lane*4;
            if(rounded) {
                for(uint j=0;j<4;j+=4)
                    *((device vec<bfloat16_t,4>*)(rounded_out+j))=vec<bfloat16_t,4>(bfloat16_t(acc[h][j+0]),bfloat16_t(acc[h][j+1]),bfloat16_t(acc[h][j+2]),bfloat16_t(acc[h][j+3]));
            } else {
                for(uint j=0;j<4;j+=4)
                    *((device float4*)(unrounded_out+j))=float4(acc[h][j+0],acc[h][j+1],acc[h][j+2],acc[h][j+3]);
            }
        }
    }
}

inline __attribute__((always_inline)) float dot16_w197_bias23(const device uchar* w,thread float* x,float s,float b,float sum) {
    return dot16(w,x,s,b,sum);
}

inline void down_all_w197_bias23(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint sg) {
    for(uint first=sg*4;first<7*2048;first+=512*4) {
        uint role=first/2048,local=first%2048;bool shared=role==6;
        Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
        const device bfloat16_t* input=x+role*512;
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[first+j]=bfloat16_t(v);}
    }
}

template<uint WIDTH>
inline void attention_merge_w182(const device bfloat16_t* q,const device bfloat16_t* k,
                      const device bfloat16_t* v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool rounded,uint lane,uint sg) {
    for(uint task=sg;task<32*(128/WIDTH);task+=608) {
        uint h=task/(128/WIDTH),dim=(task%(128/WIDTH))*WIDTH;
        float top=-3.402823466e+38f;
        for(uint p=lane;p<parts;p+=32)top=max(top,stats[(h*parts+p)*2]);
        top=simd_max(top);float z=0.0f,a[WIDTH]={};
        for(uint p=lane;p<parts;p+=32) {
            uint row=h*parts+p;float factor=metal::fast::exp(stats[row*2]-top);
            z+=factor*stats[row*2+1];
            for(uint j=0;j<WIDTH;++j) {
                uint at=partial_offset(h,p,dim+j,parts,rounded);
                a[j]+=factor*(rounded?float(((device bfloat16_t*)partial)[at]):partial[at]);
            }
        }
        z=simd_sum(z);
        float owned[2]={};
        for(uint j=0;j<WIDTH;++j) {
            float total=simd_sum(a[j]);
            if(lane==j/2)owned[j%2]=total;
        }
        if(lane<WIDTH/2)for(uint t=0;t<2;++t) {
            uint j=lane*2+t;
            out[h*128+dim+j]=bfloat16_t(z==0.0f?owned[t]:owned[t]/z);
        }
    }
}

template<uint HEADS,bool ROUNDED>
inline void attention_partial_path_w53(const device bfloat16_t* q,KVView k,
                      KVView v,const device bfloat16_t* sinks,
                      device bfloat16_t* out,device float* partial,device float* stats,
                      uint count,uint cap,uint parts,bool runtime_rounded,uint lane,uint sg) {
    constexpr bool rounded=ROUNDED;
    for(uint task=(sg/152)*((8/HEADS)*parts)+sg%152;task<(sg/152+1)*((8/HEADS)*parts);task+=152) {
        uint kv=task/((8/HEADS)*parts),part=(task/(8/HEADS))%parts,first=kv*8+(task%(8/HEADS))*HEADS;
        KVView key_base=k.head(kv,lane*4);
        KVView value_base=v.head(kv,lane*4);
        const device bfloat16_t* query_base=q+first*128+lane*4;
        float query[HEADS][4],acc[HEADS][4];
        float hi=part==0 && lane<HEADS?float(sinks[first+lane]):-3.402823466e+38f;
        float z=part==0 && lane<HEADS?1.0f:0.0f;
        for(uint h=0;h<HEADS;++h)for(uint j=0;j<4;++j) {
            query[h][j]=float(query_base[h*128+j])*0.08838834764831845f;acc[h][j]=0.0f;
        }
        uint token=part;
        if(rounded && part!=0 && token<count) {
            uint at=uint(token)*128;
            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                score=simd_sum(score);if(lane==h){hi=score;z=1.0f;}
                for(uint j=0;j<4;++j)acc[h][j]=0.0f+value[j];
            }
            token+=parts;
        }
        bfloat16_t next_key[4],next_value[4];
        if(token<count)for(uint j=0;j<4;++j) {
            uint at=uint(token)*128+j;
            next_key[j]=key_base[at];next_value[j]=value_base[at];
        }
        #pragma clang loop unroll_count(1)
        for(;token<count;token+=parts) {
            uint at=uint(token)*128;
            float key[4];bfloat16_t value[4];
            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}
            if(token+parts<count)for(uint j=0;j<4;++j) {
                uint next=at+parts*128+j;
                next_key[j]=key_base[next];next_value[j]=value_base[next];
            }
            float scores[HEADS];
            for(uint h=0;h<HEADS;++h) {
                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];
                scores[h]=simd_sum(score);
            }
            float old=0.0f,add=0.0f;
            if(lane<HEADS) {
                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));
                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;
                z=z*old+add;hi=next;
            }
            for(uint h=0;h<HEADS;++h) {
                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));
                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);
            }
        }
        if(lane<HEADS) {
            device float* stat_out=stats+(first*parts+part)*2+lane*parts*2;
            stat_out[0]=hi;stat_out[1]=z;
        }
        for(uint h=0;h<HEADS;++h) {
            device bfloat16_t* rounded_out=rounded?((device bfloat16_t*)partial)+(((first+h)*32+lane)*parts+part)*4:(device bfloat16_t*)partial;
            device float* unrounded_out=rounded?partial:partial+((first+h)*parts+part)*128+lane*4;
            if(rounded) {
                for(uint j=0;j<4;j+=4)
                    *((device vec<bfloat16_t,4>*)(rounded_out+j))=vec<bfloat16_t,4>(bfloat16_t(acc[h][j+0]),bfloat16_t(acc[h][j+1]),bfloat16_t(acc[h][j+2]),bfloat16_t(acc[h][j+3]));
            } else {
                for(uint j=0;j<4;j+=4)
                    *((device float4*)(unrounded_out+j))=float4(acc[h][j+0],acc[h][j+1],acc[h][j+2],acc[h][j+3]);
            }
        }
    }
}


template<bool LANE_SCORES>
inline void route_top_lane_scores(const device float* bias,device float* probs,
                      device uint* ids,device float* scores,uint lane,uint sg) {
    if(sg==0) {
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        float chosen=0.0f;
        for(uint rank=0;rank<6;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i>idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_max(best==peak?idx:0);
            float value=simd_shuffle(p[selected/32],selected%32);
            if(lane==0)ids[5-rank]=selected;
            if constexpr(LANE_SCORES) {
                if(lane==5-rank)chosen=value;
            } else {
                if(lane==0)scores[5-rank]=value;
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
        if constexpr(LANE_SCORES) {
            float sum=0.0f;
            for(uint j=0;j<6;++j)sum=simd_shuffle(chosen,j)+sum;
            sum+=1.0e-20f;
            if(lane<6)scores[lane]=chosen/sum;
        } else {
            if(lane==0){float sum=0.0f;for(uint j=0;j<6;++j)sum=scores[j]+sum;sum+=1.0e-20f;for(uint j=0;j<6;++j)scores[j]/=sum;}
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}
inline void rms_256_w4(const thread bfloat16_t* x,const device bfloat16_t* w,
                threadgroup bfloat16_t* out,uint tid,uint lane,uint sg,
                threadgroup float* temp,float eps) {

    float sums[2];
    for(uint leaf_set=0;leaf_set<2;++leaf_set) {
        float squared=0.0f;
        for(uint j=0;j<4;++j){float v=float(x[leaf_set*4+j]);squared+=v*v;}
        sums[leaf_set]=simd_sum(squared);
    }
    if(lane==0){temp[sg]=sums[0];temp[sg+8]=sums[1];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float leaf=lane<16?temp[lane]:0.0f;
    float total=simd_sum(leaf);
    float inverse=metal::precise::rsqrt(total/2048.0f+eps);
    for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;out[i]=w[i]*bfloat16_t(float(x[j])*inverse);}
    threadgroup_barrier(mem_flags::mem_threadgroup);

}

// Independent expert teams: two groups per routed role, four for shared.
inline void expert_team_gateup(const device uchar* bank,const threadgroup uint* d,
                      const threadgroup bfloat16_t* x,const device uint* ids,
                      const device bfloat16_t* silu,device bfloat16_t* out,
                      uint lane,uint local_sg,uint group) {
    bool shared=group>=48;
    uint role=shared?6:group/8, team=shared?group-48:group%8;
    uint teams=shared?16:8, width=shared?1024:512;
    Matrix g=weight_matrix(bank,d+(shared?45:30));
    Matrix u=weight_matrix(bank,d+(shared?50:35));
    uint expert=shared?0:ids[role];
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*2;local<width;local+=teams*8*2) {
        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};
        for(uint b=0;b<2048;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);
            for(uint j=0;j<2;++j) {
                uint row=expert*g.n+local+j,gi=row*32+scalar/64;
                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);
                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);
            }
        }
        for(uint j=0;j<2;++j) {
            float2 val=simd_sum(float2(ga[j],ua[j]));
            if(lane==0) {
                bfloat16_t g=bfloat16_t(val.x),u=bfloat16_t(val.y);
                if(shared)out[role*512+local+j]=bfloat16_t(silu[as_type<uint>(float(g))>>16]*u);
                else {auto tail=1/(1+metal::fast::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[role*512+local+j]=bfloat16_t(bfloat16_t(g*sig)*u);}
            }
        }
    }
}
inline void expert_team_down(const device uchar* bank,const threadgroup uint* d,
                     const device bfloat16_t* x,const device uint* ids,
                     device bfloat16_t* out,uint lane,uint local_sg,uint group) {
    bool shared=group>=48;
    uint role=shared?6:group/8, team=shared?group-48:group%8;
    uint teams=shared?16:8;
    Matrix m=weight_matrix(bank,d+(shared?55:40));uint expert=shared?0:ids[role];
    const device bfloat16_t* input=x+role*512;
    uint role_sg=team*8+local_sg;
    for(uint local=role_sg*4;local<2048;local+=teams*8*4) {
        float a[4]={0.0f,0.0f,0.0f,0.0f};
        for(uint b=0;b<m.k;b+=512) {
            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);
            for(uint j=0;j<4;++j) {
                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;
                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);
            }
        }
        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(v);}
    }
}

inline bool expert_team_handoff(device uint* routing,device atomic_uint* control,
                    uint tid,uint group,uint layer,uint phase,threadgroup uint* ready) {
    uint first=group>=48?48:(group/8)*8, count=group>=48?16:8;
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),layer+1,memory_order_relaxed);
        bool complete=true;
        for(uint peer=first;peer<first+count;++peer) {
            uint spins=0;
            while(atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)<layer+1 &&
                  atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
            complete=complete && atomic_load_explicit((device atomic_uint*)(routing+peer*16+14),memory_order_relaxed)>=layer+1;
        }
        ready[0]=uint(complete && atomic_load_explicit(control+2,memory_order_relaxed)==0);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    return ready[0]!=0;
}

inline void w3_route_own_rank(const device float* bias,device float* probs,
                            device float* chosen,device uint* ids,uint lane,uint sg,uint group) {
    if(sg==0 && group<48) {
        uint role=group/8;
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<=role;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            if(rank==role) {
                float value=simd_shuffle(p[selected/32],selected%32);
                if(lane==0) {
                    ids[role]=selected;
                    if(group%8==0)chosen[role]=value;
                }
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline bool kv_team_sync(device atomic_uint* control,uint tid,thread uint& phase,
                     threadgroup uint* ready,device uint* profile,uint group) {
    device atomic_uint* team_control=control+4+(group/19)*2;

    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if(tid==0) {
        uint goal=(phase+1u)*19;
        uint arrived=atomic_fetch_add_explicit(team_control,1u,memory_order_relaxed);
        uint spins=0;
        while(atomic_load_explicit(team_control,memory_order_relaxed)<goal &&
              atomic_load_explicit(control+2,memory_order_relaxed)==0 && spins<10000000) ++spins;
        if(0) {uint at=(phase*76+group)*2;profile[at]=spins;profile[at+1]=arrived-phase*76;}
        ready[0]=uint(atomic_load_explicit(team_control,memory_order_relaxed)>=goal);
        if(!ready[0])atomic_store_explicit(control+2,phase+1,memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    ++phase;
    return ready[0]!=0;

}

// Last threadgroup to leave the launch zeros the spin counters in place.
// Greedy packets reuse this buffer. The next launch must observe zeros or
// sync_all waits forever. Words 0..63 cover the grid counter, the failure
// flag, and the four KV team counters. Word 1022 is the arrival count.
// Single-token decode does not call this.
inline void release_control(device atomic_uint* control, uint tid) {
    if(tid==0) {
        uint n=atomic_fetch_add_explicit(control+1022,1u,memory_order_relaxed);
        if(n+1u==76u) {
            for(uint i=0;i<64u;++i) atomic_store_explicit(control+i,0u,memory_order_relaxed);
            atomic_store_explicit(control+1022,0u,memory_order_relaxed);
        }
    }
}

"""#
    static let bodyMax: String = #"""
#define KV_TEAM_SYNC() if(!kv_team_sync(control,tid,kv_phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}

    uint token=parameters[0], position=parameters[1], initial_position=parameters[2], layer_count=parameters[3];
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,sg=group*8+simdgroup_index_in_threadgroup;
    uint index=group*256+tid;
    device atomic_uint* control=(device atomic_uint*)private_control;
    device bfloat16_t* trace_rw=(device bfloat16_t*)trace;
    device bfloat16_t* workspace_rw=(device bfloat16_t*)workspace;
    device float* partials_rw=(device float*)partials;
    device float* statistics_rw=(device float*)statistics;
    device uint* routing_rw=(device uint*)routing;
    #define trace trace_rw
    #define workspace workspace_rw
    #define partials partials_rw
    #define statistics statistics_rw
    #define routing routing_rw
    threadgroup uint ready[1];uint phase=0;uint kv_phase=0;
    #define SYNC() if(!sync_all(control,tid,phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}


    threadgroup float reduction[33];
    threadgroup uint layer_descriptor[64];
    // Every resident group consumes scores in its private residual update.
    device uint* ids=routing+group*16;
    device float* scores=(device float*)(routing+group*16+8);
    if(tid==0)atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    bfloat16_t h[8];
    alignas(8) threadgroup bfloat16_t n[2048];
    device bfloat16_t* q=workspace+4096;
    device bfloat16_t* k=workspace+8192;
    device bfloat16_t* v=workspace+8704;
    device bfloat16_t* a=workspace+9216;
    device bfloat16_t* o=workspace+13312;
    device bfloat16_t* gate=workspace+15360;
    device bfloat16_t* up=workspace+31744;
    device bfloat16_t* act=workspace+48128;
    device bfloat16_t* down=workspace+64512;
    const device uint* final=metadata+49*64;
    const device uchar* tail=bank12;
    Matrix embedding=weight_matrix(tail,final+10);
    for(uint j=0;j<8;++j) {
        uint i=(tid+(j/4)*256)*4+j%4;
        size_t row=uint(token);uint code=embedding.w[row*2048+i];
        uint gi=row*32+i/64;
        h[j]=bfloat16_t(embedding.s[gi]*bfloat16_t(code)+embedding.b[gi]);
        if(group==0)trace[i]=h[j];
    }
    SYNC();
    for(uint layer=0;layer<uint(layer_count);++layer) {
        if(tid<64)layer_descriptor[tid]=metadata[layer*64+tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup uint* d=layer_descriptor;
        const device uchar* bank=layer<4?bank0:layer<8?bank1:layer<12?bank2:layer<16?bank3:layer<20?bank4:layer<24?bank5:layer<28?bank6:layer<32?bank7:layer<36?bank8:layer<40?bank9:layer<44?bank10:layer<48?bank11:bank12;
        bool global=d[1]!=0,sparse=d[2]!=0;
        {rms_256_w4(h,(const device bfloat16_t*)(bank+d[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
        uint cap=d[62],count=global?uint(position)+1:min(uint(position)+1,cap);
        uint insert=global?uint(position):(uint(initial_position)>=cap?(uint(position)-uint(initial_position))%cap:uint(position)%cap);
        KVView keys={kv0,kv1,kv2,kv3,d[61],cap,global};
        KVView vals=keys.value();
        // Each group owns a full Q, K or V head, including normalization and KV publication.
        for(uint work=group;work<76;work+=76) {
            uint kv_team=work/19,kv_member=work%19;
            if(kv_member>=10)continue;
            uint head=kv_member<8?kv_team*8+kv_member:kv_member==8?32+kv_team:36+kv_team;
            uint local=head<32?head:head<36?head-32:head-36;
            Matrix m=weight_matrix(bank,d+(head<32?10:head<36?15:20));
            m.w+=size_t(local)*128*1024;m.s+=local*128*32;m.b+=local*128*32;m.n=128;
            device bfloat16_t* out=(head<32?q:head<36?k:v)+local*128;
            q4_local<2048>(m,n,out,lane,simdgroup_index_in_threadgroup,8);
            threadgroup_barrier(mem_flags::mem_device);
            if(simdgroup_index_in_threadgroup==0 && head<36)
                head_norm(out,(const device bfloat16_t*)(bank+d[head<32?5:6]),1,!global,uint(position),13.287712379549449f,1e-06f,lane,0);
            threadgroup_barrier(mem_flags::mem_device);
            if(head>=32) {
                KVView target=head<36?keys:vals;
                for(uint dim=tid;dim<128;dim+=256)target.store(local,insert,dim,out[dim]);
            }
        }
        KV_TEAM_SYNC();
        bool rounded=global&&count>=1024;
        uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;
        if(group<76) {
            if(global && count>1024) {
                if(rounded) attention_partial_path_w53<8,true>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<8,false>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
            else {
                if(rounded) attention_partial_path_w53<4,true>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
                else attention_partial_path_w53<4,false>(q,keys,vals,(const device bfloat16_t*)(bank+d[7]),a,partials,statistics,count,cap,parts,rounded,lane,sg);
            }
        }
        KV_TEAM_SYNC();
        if(group<76) {if(rounded) attention_merge_vec4_w2<true>(a,partials,statistics,parts,lane,sg); else attention_merge_vec4_w2<false>(a,partials,statistics,parts,lane,sg);}
        SYNC();
        q4<4096>(weight_matrix(bank,d+25),a,o,lane,sg,608);
        SYNC();
        {
            for(uint j=0;j<8;++j){uint i=(tid+(j/4)*256)*4+j%4;h[j]=bfloat16_t(float(h[j])+float(o[i]));}
            rms_256_w4(h,(const device bfloat16_t*)(bank+d[4]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);
        }
        if(sparse) {
            if(group<76) route_dot_bf16(n,(const device bfloat16_t*)(bank+d[60]),statistics,lane,sg);
            SYNC();
            w3_route_own_rank((const device float*)(bank+d[9]),statistics,(device float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);
            // The local route_top device fence publishes this group's ids and scores.
            if(64==64) {
                if(group<64) {
                    expert_team_gateup(bank,d,n,ids,silu_table,act,lane,simdgroup_index_in_threadgroup,group);
                    if(!expert_team_handoff(routing,control,tid,group,layer,phase,ready)) {
                        if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);
                        return;
                    }
                    expert_team_down(bank,d,act,ids,down,lane,simdgroup_index_in_threadgroup,group);
                }
            } else {
                if(group<64) gateup_all(bank,d,n,ids,silu_table,act,lane,sg);
                SYNC();
                if(group<64) down_all_w197_bias23(bank,d,act,ids,down,lane,sg);
            }
            SYNC();
        } else {
            Matrix gm=weight_matrix(bank,d+30),um=weight_matrix(bank,d+35),dm=weight_matrix(bank,d+40);
            q4_local<2048>(gm,n,gate,lane,sg,608);q4_local<2048>(um,n,up,lane,sg,608);
            SYNC();
            for(uint i=index;i<gm.n;i+=19456) {
                bfloat16_t g=gate[i],u=up[i];
                auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;
                act[i]=bfloat16_t(bfloat16_t(g*sig)*u);
            }
            SYNC();
            q4<6144>(dm,act,down,lane,sg,608);
            SYNC();
        }
        {
        if(sparse) {
            float w3_chosen=lane<6?((device float*)(routing+8))[lane]:0.0f;
            float w3_sum=0.0f;
            for(uint j=0;j<6;++j)w3_sum=simd_shuffle(w3_chosen,j)+w3_sum;
            w3_sum+=1.0e-20f;
            float w3_weight=lane<6?w3_chosen/w3_sum:0.0f;
            float sums[8];
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t(down[i]*bfloat16_t(simd_shuffle(w3_weight,ushort(0)))))+0.0f;
            }
            for(uint j=0;j<8;++j)for(uint r=1;r<6;++r) {
                uint i=(tid+(j/4)*256)*4+j%4;
                sums[j]=float(bfloat16_t((float(bfloat16_t(down[r*2048+i]*bfloat16_t(simd_shuffle(w3_weight,ushort(r)))))+0.0f)+sums[j]));
            }
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=bfloat16_t(float(bfloat16_t(sums[j]))+float(down[6*2048+i]));
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        } else {
            for(uint j=0;j<8;++j) {
                uint i=(tid+(j/4)*256)*4+j%4;
                bfloat16_t mlp=down[i];
                h[j]=bfloat16_t(float(h[j])+float(mlp));
                if(group==0)trace[(layer+1)*2048+i]=h[j];
            }
        }
        }
    }
    {rms_256_w4(h,(const device bfloat16_t*)(tail+final[3]),n,tid,lane,simdgroup_index_in_threadgroup,reduction,1e-06f);}
    if(group<64) q4_local<2048>(weight_matrix(tail,final+30),n,logits,lane,sg,512);
    if(tid==0)diagnostic[group]=0;
    if(0 && tid==0 && group==0)profile[512*76*2]=phase;

"""#
}
