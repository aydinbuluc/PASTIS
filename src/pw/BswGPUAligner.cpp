#include <unistd.h>

#include <algorithm>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include "../../include/pw/BswGPUAligner.hpp"
#include "driver.hpp"

using std::min;
using std::max;
using std::string;
using std::vector;


BswGPUAligner::BswGPUAligner
(
) :
	PairwiseFunction()
{
}



void
BswGPUAligner::apply
(
    uint64_t l_col_idx, uint64_t g_col_idx,
	uint64_t l_row_idx, uint64_t g_row_idx,
	seqan::Peptide *seq_h, seqan::Peptide *seq_v,
	pastis::CommonKmers &cks, std::stringstream& ss
)
{
	
}



// @NOTE do not call this
void
BswGPUAligner::apply_batch
(
     seqan::StringSet<seqan::Gaps<seqan::Peptide>> &seqsh,
	 seqan::StringSet<seqan::Gaps<seqan::Peptide>> &seqsv,
	 uint64_t *lids,
	 uint64_t col_offset,
	 uint64_t row_offset,
	 PSpMat<pastis::CommonKmers>::Tuples &mattuples,
	 std::ofstream &afs,
	 std::ofstream &lfs
)
{
	
}



// length elimination already done before calling this
void
BswGPUAligner::apply_batch_sc
(
    seqan::StringSet<seqan::Peptide> &seqsh,
	seqan::StringSet<seqan::Peptide> &seqsv,
	uint64_t *lids,
	uint64_t col_offset,
	uint64_t row_offset,
	PSpMat<pastis::CommonKmers>::ref_tuples *mattuples,
	std::ofstream &afs,
	std::ofstream &lfs
)
{
	short sc_mat[] =
		{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,
		  0, -2, -1,  0, -4, -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3,
		 -2, -1, -1, -3, -2, -3, -1,  0, -1, -4, -2,  0,  6,  1, -3,  0,  0,  0,  1,
		 -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4, -2, -2,  1,  6,
		 -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1,
		 -4,  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2,
		 -2, -1, -3, -3, -2, -4, -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0,
		 -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4, -1,  0,  0,  2, -4,  2,  5, -2,
		  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,  0, -2,  0,
		 -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2,
		 -1, -4, -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2,
		 -2,  2, -3,  0,  0, -1, -4, -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,
		  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4, -1, -2, -3, -4, -1, -2, -3,
		 -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4, -1,  2,
		  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,
		  1, -1, -4, -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1,
		 -1, -1, -1,  1, -3, -1, -1, -4, -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0,
		 -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4, -1, -2, -2, -1, -3, -1,
		 -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,  1,
		 -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,
		  0,  0,  0, -4,  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,
		  1,  5, -2, -2,  0, -1, -1,  0, -4, -3, -3, -4, -4, -2, -2, -3, -2, -2, -3,
		 -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4, -2, -2, -2, -3, -2,
		 -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
		  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,
		  4, -3, -2, -1, -4, -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3,
		 -2,  0, -1, -4, -3, -3,  4,  1, -1, -4, -1,  0,  0,  1, -3,  3,  4, -2,  0,
		 -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,  0, -1, -1, -1,
		 -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1,
		 -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
		 -4, -4, -4, -4, -4,  1
		};


	uint64_t i, j;
	uint64_t npairs = seqan::length(seqsh);
	uint64_t npairs_elim = 0;
	static double raw_aln_time = 0.0;
	static double raw_ass_time = 0.0;

	int numThreads = 1;
	#ifdef THREADED
	#pragma omp parallel
    {
      	numThreads = omp_get_num_threads();
    }
	#endif

	vector<string> seqs_q(npairs);		// queries - shorter seqs
	vector<string> seqs_r(npairs);		// refs - longer seqs
	
	#pragma omp for
	for (uint64_t i = 0; i < npairs; ++i)
	{
		int len_seqh = seqan::length(seqsh[i]);
		int len_seqv = seqan::length(seqsv[i]);

		if (len_seqh < len_seqv)
		{
			seqan::assign(seqs_q[i], seqsh[i]);
			seqan::assign(seqs_r[i], seqsv[i]);
		}
		else
		{
			seqan::assign(seqs_q[i], seqsv[i]);
			seqan::assign(seqs_r[i], seqsh[i]);
		}
	}

	auto beg_aln = std::chrono::system_clock::now();

	// #pragma omp parallel
	// {
	// int maxnthds = omp_get_max_threads();
	// std::cout << "before ADEPT " << maxnthds << std::endl;
	// }

	gpu_bsw_driver::alignment_results res;
	kernel_driver_aa(seqs_q, seqs_r, &res, sc_mat, -11, -1);

	auto end_aln = std::chrono::system_clock::now();
	raw_aln_time += (ms_t(end_aln - beg_aln)).count();
	std::cout << "raw alignment time " << raw_aln_time << " ms" << std::endl;

	omp_set_num_threads(numThreads);

	// stats
	#pragma omp parallel
	{
		std::stringstream ss;
		#pragma omp for
		for (uint64_t i = 0; i < npairs; ++i)
		{
			// int len_seqh = seqan::length(seqsh[i]);
			// int len_seqv = seqan::length(seqsv[i]);
			int len_seqh = seqs_r[i].size();
			int len_seqv = seqs_q[i].size();
			double cov_longer = (double)(res.ref_end[i]-res.ref_begin[i]) /
				max(len_seqh, len_seqv);
			double cov_shorter = (double)(res.query_end[i]-res.query_begin[i]) /
				min(len_seqh, len_seqv);
			
			if (max(cov_longer, cov_shorter) >= 0.70) // coverage constraint
			{
				pastis::CommonKmers *cks = std::get<2>(mattuples[lids[i]]);
				cks->nrm_score = (float)(res.top_scores[i]) /
					(float)min(len_seqh, len_seqv);
				cks->score = 1;	// keep this
				// ss << (col_offset + mattuples.colindex(lids[i])) << ","
				//    << (row_offset + mattuples.rowindex(lids[i]))  << ","
				//    << res.top_scores[i] << ","
				//    << len_seqh << ","
				//    << len_seqv << ","
				//    << ((double)res.top_scores[i]/(double)len_seqh) << ","
				//    << ((double)res.top_scores[i]/(double)len_seqv) << ","
				//    << (double)(res.ref_end[i]-res.ref_begin[i]) /
				// 	max(len_seqh, len_seqv) << ","
				//    << (double)(res.query_end[i]-res.query_begin[i]) /
				// 	min(len_seqh, len_seqv) << ","
				//    << res.ref_begin[i] << ","
				//    << res.ref_end[i] << ","
				//    << res.query_begin[i] << ","
				//    << res.query_end[i]
				//    << "\n";
			}
		}

		// #pragma omp critical
		// {
		// 	afs << ss.str();
		// 	afs.flush();
		// }
	}

	return;
}
